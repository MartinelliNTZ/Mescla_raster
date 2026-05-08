# -*- coding: utf-8 -*-
"""
VALIDADOR DE RASTER GEO TIFF (DJI TERRA / ORTOMOSAICO)
------------------------------------------------------
Objetivo:
- Ler metadados completos do raster
- Detectar porque o TIFF está pesado
- Gerar checklist técnico automático
- Mostrar estimativa de memória
- Validar compressão, tiles, overviews etc
- Estatísticas radiométricas por banda
- GSD, área coberta, estimativa de RAM no QGIS
- Detecção de BigTIFF, CRS geográfico, rotação, metadados EXIF/XMP
- Benchmark de I/O de tile de amostra
- Sugestão de níveis de overview ideais
- Exportação de relatório JSON

REQUISITOS:
pip install rasterio numpy

USO:
python validar_raster.py "D:/meu_raster.tif"
python validar_raster.py "D:/meu_raster.tif" --json relatorio.json
"""

import os
import sys
import time
import math
import json
import argparse
import numpy as np
import rasterio
from rasterio.enums import Compression
from rasterio.windows import Window


# ─────────────────────────────────────────────
# UTILITÁRIOS
# ─────────────────────────────────────────────

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


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ─────────────────────────────────────────────
# ESTATÍSTICAS RADIOMÉTRICAS POR BANDA
# ─────────────────────────────────────────────

def band_statistics(ds, max_sample_pixels=10_000_000):
    """
    Lê amostra representativa de cada banda e calcula estatísticas.
    Faz downsample automático para arquivos muito grandes.
    """
    width, height, count = ds.width, ds.height, ds.count
    total_pixels = width * height

    # fator de downsample para não estourar memória
    scale = min(1.0, math.sqrt(max_sample_pixels / total_pixels))
    out_w = max(1, int(width * scale))
    out_h = max(1, int(height * scale))

    stats = []
    for band_idx in range(1, count + 1):
        data = ds.read(
            band_idx,
            out_shape=(out_h, out_w),
            resampling=rasterio.enums.Resampling.nearest
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
        dtype_max = {"uint8": 255, "uint16": 65535, "int16": 32767,
                     "uint32": 4294967295, "int32": 2147483647,
                     "float32": None, "float64": None}.get(dtype)

        vmin = float(valid.min())
        vmax = float(valid.max())
        vmean = float(valid.mean())
        vstd = float(valid.std())
        vmedian = float(np.median(valid))

        # clipping = saturação nos extremos (>0.5% dos pixels no limite)
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


# ─────────────────────────────────────────────
# BENCHMARK DE I/O (leitura de tile de amostra)
# ─────────────────────────────────────────────

def benchmark_tile_read(ds):
    """Lê um único tile do centro do raster e mede o tempo."""
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


# ─────────────────────────────────────────────
# SUGESTÃO DE NÍVEIS DE OVERVIEW
# ─────────────────────────────────────────────

def suggested_overview_levels(width, height, tile_size=256):
    """Calcula os fatores de overview ideais até que a imagem caiba num tile."""
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


# ─────────────────────────────────────────────
# DETECÇÃO DE METADADOS EXIF / XMP / TIFF TAGS
# ─────────────────────────────────────────────

def extract_metadata_tags(ds):
    meta = {}
    for ns in [None, "EXIF", "XMP", "TIFF", "IMAGE_STRUCTURE"]:
        try:
            tags = ds.tags(ns=ns) if ns else ds.tags()
            if tags:
                meta[ns or "DEFAULT"] = tags
        except Exception:
            pass
    return meta


# ─────────────────────────────────────────────
# CÁLCULO DE GSD E ÁREA
# ─────────────────────────────────────────────

def calc_gsd_and_area(ds):
    """
    Retorna GSD em cm/pixel e área coberta em hectares/km².
    Só faz sentido para CRS projetado (metros).
    """
    crs = ds.crs
    transform = ds.transform
    res_x = abs(transform.a)
    res_y = abs(transform.e)

    is_projected = crs.is_projected if crs else False

    gsd_cm = res_x * 100 if is_projected else None  # metros → cm
    area_m2 = res_x * res_y * ds.width * ds.height if is_projected else None
    area_ha = area_m2 / 10_000 if area_m2 else None
    area_km2 = area_m2 / 1_000_000 if area_m2 else None

    return gsd_cm, area_ha, area_km2, is_projected


# ─────────────────────────────────────────────
# ESTIMATIVA DE RAM NO QGIS
# ─────────────────────────────────────────────

def estimate_qgis_ram(width, height, count, bytes_per_pixel):
    """
    Estimativa conservadora de RAM para renderizar o raster em resolução total no QGIS.
    QGIS mantém em geral 2–3 cópias em cache durante renderização.
    """
    raw = width * height * count * bytes_per_pixel
    return raw * 3  # fator conservador de 3x


# ─────────────────────────────────────────────
# DETECTAR ROTAÇÃO / SKEW
# ─────────────────────────────────────────────

def has_rotation(transform):
    """Rasters com rotação têm termos b ou d != 0 no transform affine."""
    return transform.b != 0 or transform.d != 0


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main(path, json_output=None):

    if not os.path.exists(path):
        print(f"Arquivo nao encontrado: {path}")
        return

    file_size = os.path.getsize(path)
    report = {}  # para exportação JSON

    section("VALIDADOR DE RASTER — DIAGNÓSTICO COMPLETO")

    print(f"\nArquivo : {path}")
    print(f"Tamanho : {human_size(file_size)}")

    with rasterio.open(path) as ds:

        # ── Metadados básicos ──────────────────────────────────────────────
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

        predictor   = profile.get("predictor") or ds.tags(ns="IMAGE_STRUCTURE").get("PREDICTOR")
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

        raw_size           = width * height * count * bytes_per_pixel
        compression_ratio  = raw_size / file_size if file_size > 0 else 0
        megapixels         = (width * height) / 1_000_000

        # ── BigTIFF ───────────────────────────────────────────────────────
        is_bigtiff = profile.get("bigtiff", "").upper() in ("YES", "IF_SAFER", "IF_NEEDED")

        # ── GSD e Área ────────────────────────────────────────────────────
        gsd_cm, area_ha, area_km2, is_projected = calc_gsd_and_area(ds)

        # ── RAM QGIS ──────────────────────────────────────────────────────
        ram_estimate = estimate_qgis_ram(width, height, count, bytes_per_pixel)

        # ── Rotação ───────────────────────────────────────────────────────
        rotated = has_rotation(transform)

        # ── Metadados EXIF/XMP ────────────────────────────────────────────
        meta_tags = extract_metadata_tags(ds)

        # ── Overview sugerido ─────────────────────────────────────────────
        tile_sz = tile_w if tile_w else 256
        suggested_ovrs = suggested_overview_levels(width, height, tile_sz)

        # ══ SEÇÃO: METADADOS ═════════════════════════════════════════════
        section("METADADOS GERAIS")
        print(f"Driver       : {driver}")
        print(f"Largura      : {width:,} px")
        print(f"Altura       : {height:,} px")
        print(f"Bandas       : {count}")
        print(f"Megapixels   : {megapixels:.2f} MP")
        print(f"Tipo pixel   : {dtype}")
        print(f"Res X        : {res_x}")
        print(f"Res Y        : {res_y}")
        print(f"CRS          : {crs}")
        print(f"Proj. (metros): {yes_no(is_projected)}")
        print(f"NoData       : {nodata}")
        print(f"Bounds       : {bounds}")
        print(f"Rotacao/Skew : {yes_no(rotated)}")
        print(f"BigTIFF      : {yes_no(is_bigtiff)}")

        if gsd_cm is not None:
            print(f"\nGSD          : {gsd_cm:.2f} cm/pixel")
            print(f"Area coberta : {area_ha:.2f} ha  ({area_km2:.4f} km²)")
        else:
            print("\nGSD / Area   : nao calculado (CRS nao projetado ou ausente)")

        # ══ SEÇÃO: ARMAZENAMENTO ══════════════════════════════════════════
        section("ARMAZENAMENTO")
        print(f"Compressao    : {compression}")
        print(f"Predictor     : {predictor or 'NAO informado'}")
        print(f"Tiled         : {yes_no(tiled)}")
        print(f"Block shapes  : {block_shapes}")
        if tile_w and tile_h:
            print(f"Tamanho bloco : {tile_w}x{tile_h} → {tile_mem}")
        print(f"Photometric   : {photometric}")
        print(f"Overviews     : {num_overviews} nível(is) {overview_levels or 'nenhum'}")
        print(f"\nTamanho raw (sem compress.) : {human_size(raw_size)}")
        print(f"Tamanho em disco           : {human_size(file_size)}")
        print(f"Taxa de compressao         : {compression_ratio:.2f}x")
        print(f"\nRAM estimada p/ QGIS (res. total) : {human_size(ram_estimate)}")
        print(f"  (estimativa conservadora 3x raw)")

        print(f"\nOverviews sugeridos (gdaladdo) : {suggested_ovrs}")
        if overview_levels:
            missing = [o for o in suggested_ovrs if o not in overview_levels]
            if missing:
                print(f"  -> Niveis faltando              : {missing}")
            else:
                print("  -> Todos os niveis necessarios ja existem.")

        # ══ SEÇÃO: METADADOS EXIF / XMP ═══════════════════════════════════
        section("METADADOS EMBUTIDOS (EXIF / XMP / TIFF TAGS)")
        if meta_tags:
            for ns, tags in meta_tags.items():
                print(f"\n[{ns}]")
                for k, v in list(tags.items())[:30]:  # limita a 30 por namespace
                    print(f"  {k}: {v}")
        else:
            print("Nenhum metadado embutido encontrado.")

        # ══ SEÇÃO: ESTATÍSTICAS RADIOMÉTRICAS ═════════════════════════════
        section("ESTATÍSTICAS RADIOMÉTRICAS POR BANDA")
        print("(calculado sobre amostra representativa)\n")

        band_stats, scale_used = band_statistics(ds)
        if scale_used < 1.0:
            print(f"  Downsample aplicado: {scale_used*100:.1f}% da resolucao original\n")

        band_names = {1: "R/Banda1", 2: "G/Banda2", 3: "B/Banda3", 4: "A/Banda4"}
        for s in band_stats:
            b = s["band"]
            label = band_names.get(b, f"Banda{b}")
            if s["min"] is None:
                print(f"  Banda {b} ({label}): SEM DADOS VALIDOS")
                continue
            flags = []
            if s["is_constant"]:
                flags.append("BANDA CONSTANTE (morta?)")
            if s["clipping_low"]:
                flags.append("CLIPPING baixo (subexposicao?)")
            if s["clipping_high"]:
                flags.append("CLIPPING alto (saturacao?)")
            flags_str = "  *** " + " | ".join(flags) if flags else ""
            print(
                f"  Banda {b} ({label:10s}) | "
                f"Min:{s['min']:>10.2f}  Max:{s['max']:>10.2f}  "
                f"Media:{s['mean']:>10.2f}  Std:{s['std']:>10.2f}  "
                f"Mediana:{s['median']:>10.2f}  "
                f"Validos:{s['pct_valid']:>6.2f}%"
                f"{flags_str}"
            )

        # ══ BENCHMARK I/O ═════════════════════════════════════════════════
        section("BENCHMARK DE LEITURA (tile central)")
        io_time = benchmark_tile_read(ds)
        if io_time is not None:
            print(f"Tempo de leitura de 1 tile: {io_time*1000:.2f} ms")
            if io_time > 0.5:
                print("  -> LENTO (>500ms). Possível fragmentação, HDD lento ou arquivo em rede.")
            elif io_time > 0.1:
                print("  -> MODERADO (100–500ms).")
            else:
                print("  -> OK (<100ms).")
        else:
            print("Nao foi possivel realizar benchmark (raster nao tiled).")

        # ══ CHECKLIST DE VALIDAÇÃO ════════════════════════════════════════
        section("CHECKLIST DE VALIDAÇÃO")
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
            problems.append(f"Overviews incompletos. Niveis faltando: {[o for o in suggested_ovrs if o not in overview_levels]}")
        if predictor is None and any(x in compression.lower() for x in ["deflate", "lzw"]):
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

        if len(problems) == 0:
            print("Nenhum problema grave detectado. ✓")
        else:
            for i, p in enumerate(problems, 1):
                print(f"{i:2d}. {p}")

        # ══ RECOMENDAÇÕES / GDAL ══════════════════════════════════════════
        section("COMANDO GDAL RECOMENDADO")

        ovr_factors = " ".join(str(o) for o in suggested_ovrs) if suggested_ovrs else "2 4 8 16 32"
        out_path = path[:-4] + "_OTIMIZADO.tif"

        print(rf"""
gdal_translate "{path}" "{out_path}" ^
    -co COMPRESS=LZW ^
    -co PREDICTOR=2 ^
    -co TILED=YES ^
    -co BLOCKXSIZE=512 ^
    -co BLOCKYSIZE=512 ^
    -co BIGTIFF=YES

gdaladdo -r average "{out_path}" {ovr_factors}
""")

        if dtype == "uint16" and count >= 3:
            print("SUGESTÃO EXTRA — Converter RGB UINT16 → UINT8 (reduz ~50% do tamanho):")
            print(f"""
gdal_translate "{path}" "{out_path}" ^
    -ot Byte ^
    -scale ^
    -co COMPRESS=LZW ^
    -co PREDICTOR=2 ^
    -co TILED=YES ^
    -co BIGTIFF=YES
""")

        # ══ EXPORTAÇÃO JSON ═══════════════════════════════════════════════
        report = {
            "arquivo": path,
            "tamanho_disco": file_size,
            "tamanho_disco_human": human_size(file_size),
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
        }

        if json_output:
            with open(json_output, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            print(f"\nRelatório JSON exportado: {json_output}")

    print("\nFINALIZADO.")
    return report


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Validador de Raster GeoTIFF")
    parser.add_argument("raster", nargs="?", help="Caminho para o arquivo .tif")
    parser.add_argument("--json", metavar="ARQUIVO.json", help="Exportar relatório JSON")
    args = parser.parse_args()

    if args.raster:
        main(args.raster, json_output=args.json)
        sys.exit(0)

    # busca automática na pasta ./A
    source_dir = os.path.join(".", "A")
    if not os.path.isdir(source_dir):
        print("Nenhum arquivo informado e pasta ./A nao encontrada.")
        print("Uso: python validar_raster.py caminho/para/raster.tif")
        sys.exit(1)

    tif_files = [
        os.path.join(source_dir, f)
        for f in os.listdir(source_dir)
        if f.lower().endswith((".tif", ".tiff"))
    ]

    if len(tif_files) == 0:
        print("Nenhum TIFF encontrado na pasta ./A.")
    elif len(tif_files) == 1:
        print(f"Raster encontrado automaticamente: {tif_files[0]}")
        main(tif_files[0], json_output=args.json)
    else:
        print("Multiplos TIFFs encontrados:\n")
        for i, f in enumerate(tif_files):
            print(f"  [{i}] {os.path.basename(f)}")
        idx = int(input("\nEscolha o indice: "))
        main(tif_files[idx], json_output=args.json)
        