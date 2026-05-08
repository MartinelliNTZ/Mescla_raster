# -*- coding: utf-8 -*-
"""
MOSAICO MERGE + DIAGNÓSTICO
============================
Script que:
1) Varre a pasta INPUT em busca de arquivos .tif
2) Gera diagnóstico JSON de cada arquivo de entrada
3) Executa o mosaico feathering (mesclagem com suavização)
4) Gera diagnóstico JSON do arquivo de saída
5) Salva tudo na pasta OUTPUT

BASEADO EM:
  - a2.py  → mosaico feathering
  - d2.py  → validador/diagnóstico de raster
"""

import os
import sys
import time
import math
import json
import glob
import warnings
import numpy as np
from scipy.ndimage import distance_transform_edt
import rasterio
from rasterio.warp import reproject, Resampling as WarpResampling
from rasterio.merge import merge
from rasterio.enums import ColorInterp, Resampling as RasterioResampling
from rasterio.windows import Window

warnings.filterwarnings("ignore")

# ╔══════════════════════════════════════════════════════════════╗
# ║         CONFIGURAÇÕES — VARIÁVEIS CONFIGURÁVEIS            ║
# ║   Altere aqui os caminhos e parâmetros conforme necessário ║
# ╚══════════════════════════════════════════════════════════════╝

# ── Diretórios de entrada e saída ──────────────────────────────
INPUT_DIR  = r"input"          # Pasta com os TIFFs de entrada
OUTPUT_DIR = r"output"         # Pasta onde serão salvos os resultados

# ── Nome do arquivo de saída (mosaico final) ───────────────────
OUTPUT_FILENAME = "mosaico_final.tif"

# ── Extensão dos arquivos de entrada ───────────────────────────
INPUT_PATTERN = "*.tif"

# ── Parâmetros de compressão do mosaico de saída ───────────────
OUTPUT_COMPRESS   = "deflate"
OUTPUT_PREDICTOR  = 2
OUTPUT_TILED      = True
OUTPUT_BLOCK_X    = 512
OUTPUT_BLOCK_Y    = 512
OUTPUT_INTERLEAVE = "pixel"
OUTPUT_NODATA     = None       # None = sem nodata (transparência pelo alpha)

# ── Níveis de overview para o mosaico de saída ─────────────────
OVERVIEW_LEVELS = [2, 4, 8, 16, 32, 64]
OVERVIEW_RESAMPLING = WarpResampling.nearest

# ── Amostragem para estatísticas do diagnóstico ────────────────
DIAG_MAX_SAMPLE_PIXELS = 10_000_000

# ── Flags de execução ──────────────────────────────────────────
RUN_DIAGNOSTIC_INPUTS  = True   # Gerar JSON de diagnóstico para cada entrada
RUN_DIAGNOSTIC_OUTPUT  = True   # Gerar JSON de diagnóstico para a saída
RUN_MOSAIC             = True   # Executar o mosaico (mesclagem)

# ╔══════════════════════════════════════════════════════════════╗
# ║              FUNÇÕES DE DIAGNÓSTICO (d2.py)                 ║
# ╚══════════════════════════════════════════════════════════════╝


def human_size(size):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


def compression_name(comp):
    if comp is None:
        return "NONE"
    return str(comp)


def yes_no(v):
    return "SIM" if v else "NAO"


def band_statistics(ds, max_sample_pixels=DIAG_MAX_SAMPLE_PIXELS):
    """Calcula estatísticas radiométricas por banda com downsampling automático."""
    width, height, count = ds.width, ds.height, ds.count
    total_pixels = width * height

    scale = min(1.0, math.sqrt(max_sample_pixels / total_pixels))
    out_w = max(1, int(width * scale))
    out_h = max(1, int(height * scale))

    stats = []
    for band_idx in range(1, count + 1):
        data = ds.read(
            band_idx,
            out_shape=(out_h, out_w),
            resampling=RasterioResampling.nearest
        ).astype("float32")

        nodata = ds.nodata
        if nodata is not None:
            mask = data != nodata
        else:
            mask = np.ones_like(data, dtype=bool)

        valid = data[mask]
        pct_valid = (mask.sum() / data.size) * 100

        if valid.size == 0:
            stats.append({
                "band": band_idx,
                "min": None, "max": None, "mean": None,
                "std": None, "median": None,
                "pct_valid": 0.0,
                "is_constant": True,
                "clipping_low": False,
                "clipping_high": False,
            })
            continue

        dtype = ds.dtypes[band_idx - 1]
        dtype_max_map = {"uint8": 255, "uint16": 65535, "int16": 32767,
                         "uint32": 4294967295, "int32": 2147483647,
                         "float32": None, "float64": None}
        dtype_max = dtype_max_map.get(dtype)

        vmin = float(valid.min())
        vmax = float(valid.max())
        vmean = float(valid.mean())
        vstd = float(valid.std())
        vmedian = float(np.median(valid))

        clip_low = False
        clip_high = False
        if dtype_max is not None:
            dtype_info = np.iinfo(np.dtype(dtype))
            low_limit = dtype_info.min
            clip_low = float((valid == low_limit).sum() / valid.size) > 0.005
            clip_high = float((valid == dtype_max).sum() / valid.size) > 0.005

        stats.append({
            "band": band_idx,
            "min": round(vmin, 4),
            "max": round(vmax, 4),
            "mean": round(vmean, 4),
            "std": round(vstd, 4),
            "median": round(vmedian, 4),
            "pct_valid": round(pct_valid, 2),
            "is_constant": vmin == vmax,
            "clipping_low": clip_low,
            "clipping_high": clip_high,
        })

    return stats, scale


def benchmark_tile_read(ds):
    """Mede tempo de leitura de um tile central."""
    block_shapes = ds.block_shapes
    if not block_shapes or block_shapes[0][0] is None:
        return None

    tile_h, tile_w = block_shapes[0]
    cx = ds.width // 2
    cy = ds.height // 2

    col_off = max(0, cx - tile_w // 2)
    row_off = max(0, cy - tile_h // 2)
    w = min(tile_w, ds.width - col_off)
    h = min(tile_h, ds.height - row_off)

    win = Window(col_off, row_off, w, h)

    t0 = time.perf_counter()
    ds.read(1, window=win)
    elapsed = time.perf_counter() - t0

    return elapsed


def suggested_overview_levels(width, height, tile_size=256):
    """Calcula níveis de overview ideais."""
    levels = []
    factor = 2
    w, h = width, height
    while w > tile_size or h > tile_size:
        levels.append(factor)
        w //= 2
        h //= 2
        factor *= 2
        if factor > 1024:
            break
    return levels


def extract_metadata_tags(ds):
    """Extrai metadados EXIF/XMP/TIFF."""
    meta = {}
    for ns in [None, "EXIF", "XMP", "TIFF", "IMAGE_STRUCTURE"]:
        try:
            tags = ds.tags(ns=ns) if ns else ds.tags()
            if tags:
                meta[ns or "DEFAULT"] = tags
        except Exception:
            pass
    return meta


def calc_gsd_and_area(ds):
    """Calcula GSD em cm/pixel e área."""
    crs = ds.crs
    transform = ds.transform
    res_x = abs(transform.a)
    res_y = abs(transform.e)

    is_projected = crs.is_projected if crs else False

    gsd_cm = res_x * 100 if is_projected else None
    area_m2 = res_x * res_y * ds.width * ds.height if is_projected else None
    area_ha = area_m2 / 10_000 if area_m2 else None
    area_km2 = area_m2 / 1_000_000 if area_m2 else None

    return gsd_cm, area_ha, area_km2, is_projected


def estimate_qgis_ram(width, height, count, bytes_per_pixel):
    """RAM estimada para QGIS renderizar em resolução total."""
    raw = width * height * count * bytes_per_pixel
    return raw * 3


def has_rotation(transform):
    """Detecta rotação/skew."""
    return transform.b != 0 or transform.d != 0


def diagnostic_report(path):
    """
    Gera relatório de diagnóstico completo para um arquivo raster.
    Retorna um dicionário com todos os dados (pronto para JSON).
    """
    if not os.path.exists(path):
        print(f"  [DIAG] Arquivo não encontrado: {path}")
        return {"erro": "Arquivo não encontrado", "arquivo": path}

    file_size = os.path.getsize(path)
    report = {
        "arquivo": os.path.abspath(path),
        "tamanho_disco": file_size,
        "tamanho_disco_human": human_size(file_size),
    }

    try:
        with rasterio.open(path) as ds:
            width     = ds.width
            height    = ds.height
            count     = ds.count
            dtypes    = ds.dtypes
            crs       = ds.crs
            transform = ds.transform
            nodata    = ds.nodata
            bounds    = ds.bounds
            res_x     = abs(transform.a)
            res_y     = abs(transform.e)
            driver    = ds.driver

            compression  = compression_name(ds.compression)
            tiled        = ds.is_tiled
            block_shapes = ds.block_shapes
            profile      = ds.profile

            predictor = profile.get("predictor") or ds.tags(ns="IMAGE_STRUCTURE").get("PREDICTOR")
            photometric = ds.tags(ns="TIFF").get("PHOTOMETRIC") or ds.tags().get("PHOTOMETRIC")

            overview_levels = ds.overviews(1)
            num_overviews   = len(overview_levels)
            has_alpha       = count == 4

            dtype = dtypes[0]
            bytes_per_pixel = {
                "uint8": 1, "int8": 1,
                "uint16": 2, "int16": 2,
                "uint32": 4, "int32": 4,
                "float32": 4, "float64": 8,
            }.get(dtype, 4)

            tile_w, tile_h = block_shapes[0] if block_shapes else (None, None)
            tile_bytes = (tile_w * tile_h * count * bytes_per_pixel) if tile_w and tile_h else 0
            tile_mem   = human_size(tile_bytes)

            raw_size          = width * height * count * bytes_per_pixel
            compression_ratio = raw_size / file_size if file_size > 0 else 0
            megapixels        = (width * height) / 1_000_000

            is_bigtiff = profile.get("bigtiff", "").upper() in ("YES", "IF_SAFER", "IF_NEEDED")

            gsd_cm, area_ha, area_km2, is_projected = calc_gsd_and_area(ds)

            ram_estimate = estimate_qgis_ram(width, height, count, bytes_per_pixel)

            rotated = has_rotation(transform)

            meta_tags = extract_metadata_tags(ds)

            tile_sz = tile_w if tile_w else 256
            suggested_ovrs = suggested_overview_levels(width, height, tile_sz)

            band_stats, scale_used = band_statistics(ds)

            io_time = benchmark_tile_read(ds)

            # ── Checklist de problemas ──────────────────────────────────
            problems = []

            if dtype == "float32":
                problems.append("Raster FLOAT32 detectado → MUITO pesado. Ortomosaicos RGB normalmente deveriam ser UINT8.")
            if dtype == "float64":
                problems.append("Raster FLOAT64 detectado → extremamente pesado.")
            if compression in ["NONE", "None"]:
                problems.append("Raster SEM compressao.")
            if not tiled:
                problems.append("Raster nao esta tiled (strip layout). Leitura aleatoria é lenta.")
            if count >= 3 and dtype == "uint16":
                problems.append("RGB em UINT16 → DJI Terra costuma exportar pesado assim. Converter para UINT8 pode reduzir 50% do tamanho.")
            if file_size > 20 * 1024**3:
                problems.append("Arquivo acima de 20 GB.")
            if num_overviews == 0:
                problems.append("Sem overviews internos. QGIS processará resolucao total para qualquer zoom → muito lento.")
            elif suggested_ovrs and any(o not in overview_levels for o in suggested_ovrs):
                missing = [o for o in suggested_ovrs if o not in overview_levels]
                problems.append(f"Overviews incompletos. Niveis faltando: {missing}")
            if predictor is None and any(x in str(compression).lower() for x in ["deflate", "lzw"]):
                problems.append("Compressao sem PREDICTOR configurado. Usar PREDICTOR=2 melhora compressao e velocidade.")
            if has_alpha:
                problems.append("4 bandas (inclui alpha) → pode reduzir performance de renderizacao no QGIS.")
            if res_x < 0.01:
                problems.append("Resolucao espacial extremamente alta.")
            if not is_projected:
                problems.append("CRS nao projetado (geografico / graus). Para mapeamento, use CRS em metros (ex: SIRGAS 2000 UTM).")
            if rotated:
                problems.append("Raster com rotacao/skew detectado. Alguns softwares nao suportam bem.")
            if not is_bigtiff and raw_size > 4 * 1024**3:
                problems.append("Arquivo >4GB sem BigTIFF. Pode corromper em ferramentas GDAL mais antigas.")
            if ram_estimate > 16 * 1024**3:
                problems.append(f"RAM estimada para QGIS em resolucao total: {human_size(ram_estimate)}. Maquinas com <16GB RAM vao travar.")

            for s in band_stats:
                if s["is_constant"]:
                    problems.append(f"Banda {s['band']} com valor constante (pode estar morta ou vazia).")
                if s["clipping_high"]:
                    problems.append(f"Banda {s['band']} com saturacao (clipping alto). Verifique exposicao.")
                if s["clipping_low"]:
                    problems.append(f"Banda {s['band']} com subexposicao (clipping baixo).")
                if s["pct_valid"] < 50:
                    problems.append(f"Banda {s['band']}: apenas {s['pct_valid']:.1f}% pixels validos. Alto percentual de NoData.")

            report.update({
                "driver": driver,
                "largura": width,
                "altura": height,
                "bandas": count,
                "megapixels": round(megapixels, 2),
                "dtype": dtype,
                "crs": str(crs),
                "is_projected": is_projected,
                "res_x": res_x,
                "res_y": res_y,
                "gsd_cm": round(gsd_cm, 4) if gsd_cm else None,
                "area_ha": round(area_ha, 4) if area_ha else None,
                "area_km2": round(area_km2, 6) if area_km2 else None,
                "nodata": nodata,
                "compressao": compression,
                "predictor": predictor,
                "tiled": tiled,
                "block_shapes": [list(b) for b in block_shapes] if block_shapes else [],
                "photometric": photometric,
                "overviews": overview_levels,
                "overviews_sugeridos": suggested_ovrs,
                "bigtiff": is_bigtiff,
                "rotacao": rotated,
                "has_alpha": has_alpha,
                "tamanho_raw": raw_size,
                "taxa_compressao": round(compression_ratio, 2),
                "ram_estimada_qgis": ram_estimate,
                "ram_estimada_qgis_human": human_size(ram_estimate),
                "benchmark_tile_ms": round(io_time * 1000, 2) if io_time else None,
                "band_stats": band_stats,
                "meta_tags": {k: dict(v) for k, v in meta_tags.items()},
                "problemas": problems,
                "diagnostico_ok": True,
            })

    except Exception as e:
        report["erro"] = str(e)
        report["diagnostico_ok"] = False

    return report


def save_diagnostic_json(report, output_path):
    """Salva relatório de diagnóstico em JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"  [DIAG] Diagnóstico salvo: {output_path}")


# ╔══════════════════════════════════════════════════════════════╗
# ║            FUNÇÃO DE MOSAICO FEATHERING (a2.py)             ║
# ╚══════════════════════════════════════════════════════════════╝


def run_mosaic(input_files, output_path):
    """
    Executa o mosaico feathering full-RAM (float32).
    Parâmetros:
        input_files  : lista de caminhos completos para arquivos .tif
        output_path  : caminho completo para o arquivo de saída .tif
    Retorna:
        output_path em caso de sucesso, None em caso de erro.
    """
    print(f"\n{'='*60}")
    print("MOSAICO FEATHERING FULL-RAM (float32)")
    print(f"{'='*60}")
    print(f"  Arquivos de entrada: {len(input_files)}")
    for f in input_files:
        print(f"    - {f}")
    print(f"  Saída: {output_path}")

    # ── 1. Abrir datasets ──────────────────────────────────────────────────
    datasets = [rasterio.open(f) for f in input_files]

    # ── 2. Grid de referência via merge ────────────────────────────────────
    print("\n  [MOSAICO] Criando grid de referência...")
    mosaic_ref, out_transform = merge(datasets, method='first')
    out_crs   = datasets[0].crs
    out_dtype = datasets[0].dtypes[0]
    _, height, width = mosaic_ref.shape
    del mosaic_ref

    print(f"  [MOSAICO] Grid: {width}x{height}  dtype={out_dtype}")
    iinfo = np.iinfo(out_dtype)

    # ── 3. Acumuladores float32 ────────────────────────────────────────────
    accum_rgb   = np.zeros((3, height, width), dtype=np.float32)
    accum_alpha = np.zeros((height, width),    dtype=np.float32)
    accum_w     = np.zeros((height, width),    dtype=np.float32)

    # ── 4. Processar cada imagem ───────────────────────────────────────────
    for ds in datasets:
        name = os.path.basename(ds.name)
        print(f"\n  [MOSAICO] Processando {name}...")

        # Máscara via dataset_mask
        alpha_orig = ds.dataset_mask().astype(np.float32)
        mask = alpha_orig > 0
        dist = distance_transform_edt(mask).astype(np.float32)
        dmax = dist.max()
        if dmax > 0:
            dist /= dmax
        del alpha_orig, mask

        # Reprojetar peso
        peso = np.zeros((height, width), dtype=np.float32)
        reproject(
            source=dist,
            destination=peso,
            src_transform=ds.transform,
            src_crs=ds.crs,
            dst_transform=out_transform,
            dst_crs=out_crs,
            resampling=WarpResampling.bilinear,
        )
        del dist
        print(f"    peso max={peso.max():.3f}")

        # Ler e reprojetar dados
        src_data = ds.read().astype(np.float32)
        dst_data = np.zeros((4, height, width), dtype=np.float32)
        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=ds.transform,
            src_crs=ds.crs,
            dst_transform=out_transform,
            dst_crs=out_crs,
            resampling=WarpResampling.bilinear,
        )
        del src_data

        accum_rgb   += dst_data[:3] * peso
        accum_alpha += dst_data[3]  * peso
        accum_w     += peso
        del dst_data, peso
        print(f"    [MOSAICO] OK")

    # ── 5. Normalizar ──────────────────────────────────────────────────────
    print("\n  [MOSAICO] Normalizando...")
    valido = accum_w > 0

    resultado = np.zeros((4, height, width), dtype=out_dtype)
    tmp = np.zeros((height, width), dtype=np.float32)

    for c in range(3):
        np.divide(accum_rgb[c], accum_w, out=tmp, where=valido)
        resultado[c] = np.clip(tmp, iinfo.min, iinfo.max).astype(out_dtype)

    np.divide(accum_alpha, accum_w, out=tmp, where=valido)
    resultado[3] = np.clip(tmp, iinfo.min, iinfo.max).astype(out_dtype)

    del accum_rgb, accum_alpha, accum_w, valido, tmp

    # ── 6. Salvar ──────────────────────────────────────────────────────────
    print(f"\n  [MOSAICO] Salvando: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=4,
        dtype=out_dtype,
        crs=out_crs,
        transform=out_transform,
        compress=OUTPUT_COMPRESS,
        predictor=OUTPUT_PREDICTOR,
        tiled=OUTPUT_TILED,
        blockxsize=OUTPUT_BLOCK_X,
        blockysize=OUTPUT_BLOCK_Y,
        interleave=OUTPUT_INTERLEAVE,
        nodata=OUTPUT_NODATA,
    ) as dst:
        dst.write(resultado)
        dst.colorinterp = (
            ColorInterp.red,
            ColorInterp.green,
            ColorInterp.blue,
            ColorInterp.alpha,
        )

    del resultado

    # ── 7. Overviews ───────────────────────────────────────────────────────
    print("  [MOSAICO] Gerando overviews...")
    with rasterio.open(output_path, 'r+') as dst:
        dst.build_overviews(OVERVIEW_LEVELS, OVERVIEW_RESAMPLING)
        dst.update_tags(ns='rio_overview', resampling='nearest')

    # ── Fechar datasets ────────────────────────────────────────────────────
    for ds in datasets:
        ds.close()

    print(f"\n  [MOSAICO] Mosaico concluído: {output_path}")
    return output_path


# ╔══════════════════════════════════════════════════════════════╗
# ║                      FUNÇÃO PRINCIPAL                       ║
# ╚══════════════════════════════════════════════════════════════╝


def main():
    print("=" * 70)
    print(" MOSAICO MERGE + DIAGNÓSTICO")
    print("=" * 70)
    print(f"\n  INPUT_DIR  : {os.path.abspath(INPUT_DIR)}")
    print(f"  OUTPUT_DIR : {os.path.abspath(OUTPUT_DIR)}")
    print(f"  OUTPUT_FILE: {OUTPUT_FILENAME}")

    # ── Criar diretório de saída ───────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Encontrar arquivos de entrada ───────────────────────────────────
    input_pattern = os.path.join(INPUT_DIR, INPUT_PATTERN)
    input_files = glob.glob(input_pattern)
    input_files = [f for f in input_files if OUTPUT_FILENAME not in os.path.basename(f)]
    input_files = sorted(input_files)

    if len(input_files) == 0:
        print(f"\n  ERRO: Nenhum arquivo '{INPUT_PATTERN}' encontrado em: {os.path.abspath(INPUT_DIR)}")
        sys.exit(1)

    print(f"\n  Arquivos encontrados: {len(input_files)}")
    for i, f in enumerate(input_files, 1):
        print(f"    [{i}] {os.path.basename(f)}")

    # ── 2. Gerar diagnóstico de cada arquivo de entrada ────────────────────
    if RUN_DIAGNOSTIC_INPUTS:
        print(f"\n{'='*70}")
        print(" DIAGNÓSTICO DOS ARQUIVOS DE ENTRADA")
        print(f"{'='*70}")

        for input_file in input_files:
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            json_path = os.path.join(OUTPUT_DIR, f"diagnostico_input_{base_name}.json")
            print(f"\n  Processando diagnóstico de: {os.path.basename(input_file)}")
            report = diagnostic_report(input_file)
            save_diagnostic_json(report, json_path)

    # ── 3. Executar mosaico ────────────────────────────────────────────────
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    if RUN_MOSAIC:
        if os.path.exists(output_path):
            print(f"\n  ATENÇÃO: '{output_path}' já existe e será sobrescrito.")
        run_mosaic(input_files, output_path)
    else:
        if not os.path.exists(output_path):
            print(f"\n  AVISO: mosaico desabilitado e arquivo de saída não existe: {output_path}")
            sys.exit(1)
        print(f"\n  Pulando execução do mosaico (RUN_MOSAIC=False). Usando arquivo existente.")

    # ── 4. Gerar diagnóstico do arquivo de saída ───────────────────────────
    if RUN_DIAGNOSTIC_OUTPUT and os.path.exists(output_path):
        print(f"\n{'='*70}")
        print(" DIAGNÓSTICO DO ARQUIVO DE SAÍDA")
        print(f"{'='*70}")

        json_output_path = os.path.join(OUTPUT_DIR, "diagnostico_output.json")
        print(f"\n  Processando diagnóstico de: {OUTPUT_FILENAME}")
        report = diagnostic_report(output_path)
        save_diagnostic_json(report, json_output_path)

        # ── Exibir resumo do diagnóstico na tela ───────────────────────────
        print(f"\n  {'='*50}")
        print(f"  RESUMO DO MOSAICO FINAL")
        print(f"  {'='*50}")
        print(f"  Arquivo     : {output_path}")
        print(f"  Tamanho     : {human_size(report.get('tamanho_disco', 0))}")
        print(f"  Dimensões   : {report.get('largura', '?')} x {report.get('altura', '?')} px")
        print(f"  Bandas      : {report.get('bandas', '?')}")
        print(f"  Resolução   : {report.get('res_x', '?'):.4f} x {report.get('res_y', '?'):.4f} m/px")
        if report.get('gsd_cm'):
            print(f"  GSD         : {report['gsd_cm']:.2f} cm/px")
        if report.get('area_ha'):
            print(f"  Área        : {report['area_ha']:.2f} ha")
        if report.get('compressao'):
            print(f"  Compressão  : {report['compressao']}")
        if report.get('taxa_compressao'):
            print(f"  Taxa compr. : {report['taxa_compressao']:.2f}x")
        problemas = report.get('problemas', [])
        if problemas:
            print(f"  Problemas   : {len(problemas)}")
            for p in problemas[:5]:
                print(f"    - {p}")
            if len(problemas) > 5:
                print(f"    ... e mais {len(problemas)-5} problema(s) (veja JSON completo)")
        else:
            print(f"  Problemas   : Nenhum detectado ✓")
        print(f"  {'='*50}")

    # ── 5. Listar arquivos gerados ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print(" ARQUIVOS GERADOS NA PASTA OUTPUT")
    print(f"{'='*70}")
    output_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*")))
    for f in output_files:
        size = human_size(os.path.getsize(f))
        print(f"  {size:>8}  {os.path.basename(f)}")

    print(f"\n{'='*70}")
    print(" PROCESSO CONCLUÍDO COM SUCESSO!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()