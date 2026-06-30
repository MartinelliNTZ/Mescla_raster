# -*- coding: utf-8 -*-
"""
MOSAICO MERGE + DIAGNÓSTICO (com suporte a .ECW)
=================================================
Script que:
1) Varre a pasta INPUT em busca de arquivos .tif e .ecw
2) Gera diagnóstico JSON de cada arquivo de entrada
3) Executa o mosaico feathering (mesclagem com suavização)
4) Gera diagnóstico JSON do arquivo de saída
5) Salva tudo na pasta OUTPUT

Suporta:
  - Entrada: GeoTIFF (.tif, .tiff) e ECW (.ecw)
  - Saída:   GeoTIFF (.tif) ou ECW (.ecw)

Para ECW, usa o gdal_translate do QGIS com o plugin ECW que já está instalado.
"""

import os
import sys
import time
import math
import json
import glob
import warnings
import subprocess
import tempfile
import shutil
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
# ╚══════════════════════════════════════════════════════════════╝

# ── SEUS DIRETÓRIOS (NÃO MEXER) ────────────────────────────────
INPUT_DIR  = r"G:\BR070\ETAPA2\OUTPUT\ORTOMOSAICO_FINAL_01\ecw"
OUTPUT_DIR = r"G:\BR070\ETAPA2\OUTPUT\ORTOMOSAICO_FINAL_01\ecw"

# ── Nome do arquivo de saída ───────────────────────────────────
OUTPUT_FILENAME = "mosaico_final.ecw"

# ── Extensões dos arquivos de entrada ──────────────────────────
INPUT_EXTENSIONS = ["*.tif", "*.tiff", "*.ecw"]

# ── Parâmetros de compressão do mosaico de saída (TIFF) ────────
OUTPUT_COMPRESS   = "deflate"
OUTPUT_PREDICTOR  = 2
OUTPUT_TILED      = True
OUTPUT_BLOCK_X    = 512
OUTPUT_BLOCK_Y    = 512
OUTPUT_INTERLEAVE = "pixel"
OUTPUT_NODATA     = None

# ── Parâmetros para conversão ECW (qualidade) ──────────────────
ECW_TARGET_PERCENT = 90

# ── Níveis de overview ─────────────────────────────────────────
OVERVIEW_LEVELS = [2, 4, 8, 16, 32, 64, 128, 256]
OVERVIEW_RESAMPLING = WarpResampling.nearest
DIAG_MAX_SAMPLE_PIXELS = 10_000_000

# ── Flags de execução ──────────────────────────────────────────
RUN_DIAGNOSTIC_INPUTS  = True
RUN_DIAGNOSTIC_OUTPUT  = True
RUN_MOSAIC             = True
KEEP_TEMP_TIFFS        = False

# ╔══════════════════════════════════════════════════════════════╗
# ║       CONFIGURAÇÃO GDAL — QGIS + PLUGIN ECW                ║
# ╚══════════════════════════════════════════════════════════════╝

# O QGIS tem o plugin ECW mas o gdal_translate sozinho não acha.
# Vamos configurar manualmente:

QGIS_VERSIONS = [
    {
        "path": r"C:\Program Files\QGIS 3.16",
        "gdal_plugins": r"bin\gdalplugins",
        "bin": r"bin\gdal_translate.exe",
    },
    {
        "path": r"C:\Program Files\QGIS 3.34.12",
        "gdal_plugins": r"apps\gdal\lib\gdalplugins",
        "bin": r"bin\gdal_translate.exe",
    },
    {
        "path": r"C:\Program Files\QGIS 3.40.14",
        "gdal_plugins": r"apps\gdal\lib\gdalplugins",
        "bin": r"bin\gdal_translate.exe",
    },
    {
        "path": r"C:\Program Files\QGIS 4.0.0",
        "gdal_plugins": r"apps\gdal\lib\gdalplugins",
        "bin": r"bin\gdal_translate.exe",
    },
]

def setup_gdal_environment():
    """
    Configura as variáveis de ambiente para usar o gdal_translate do QGIS
    com o plugin ECW. Retorna (gdal_translate_path, env_dict) ou (None, None).
    """
    for qver in QGIS_VERSIONS:
        gdal_bin = os.path.join(qver["path"], qver["bin"])
        plugin_dir = os.path.join(qver["path"], qver["gdal_plugins"])
        
        # Verificar se existe o binário E o plugin ECW
        ecw_plugin = os.path.join(plugin_dir, "gdal_ECW_JP2ECW.dll")
        if os.path.exists(gdal_bin) and os.path.exists(plugin_dir):
            # Verificar se tem algum plugin ECW na pasta
            has_ecw = os.path.exists(ecw_plugin) or any("ECW" in f for f in os.listdir(plugin_dir))
            if has_ecw:
                # Criar environment com GDAL_DRIVER_PATH
                env = os.environ.copy()
                env["GDAL_DRIVER_PATH"] = plugin_dir
                
                # Testar se funciona
                try:
                    result = subprocess.run(
                        [gdal_bin, "--version"],
                        capture_output=True, text=True, timeout=10,
                        env=env
                    )
                    if result.returncode == 0:
                        print(f"  ✓ GDAL encontrado: {qver['path']}")
                        print(f"  ✓ Plugin ECW em: {plugin_dir}")
                        return gdal_bin, env
                except Exception:
                    pass
    
    return None, None

# ╔══════════════════════════════════════════════════════════════╗
# ║              FUNÇÕES AUXILIARES                             ║
# ╚══════════════════════════════════════════════════════════════╝


def human_size(size):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


def run_gdal(cmd, env, timeout=7200):
    """Executa um comando gdal_translate com environment configurado."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        if result.returncode != 0:
            return False, result.stderr[:500]
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def convert_ecw_to_tiff(ecw_path, tiff_path, gdal_cmd, env):
    """Converte ECW → TIFF via gdal_translate com plugin ECW."""
    cmd = [gdal_cmd, '-of', 'GTiff',
           '-co', 'COMPRESS=DEFLATE', '-co', 'PREDICTOR=2',
           '-co', 'TILED=YES', '-co', 'BIGTIFF=IF_SAFER',
           ecw_path, tiff_path]
    print(f"    [CONV] ECW → TIFF: {os.path.basename(ecw_path)}")
    ok, err = run_gdal(cmd, env, timeout=7200)
    if ok:
        print(f"    [CONV] OK")
        return True
    else:
        print(f"    [CONV] ERRO: {err}")
        return False


def convert_tiff_to_ecw(tiff_path, ecw_path, gdal_cmd, env):
    """Converte TIFF → ECW via gdal_translate com plugin ECW."""
    cmd = [gdal_cmd, '-of', 'ECW',
           '-co', f'TARGET={ECW_TARGET_PERCENT}',
           '-co', 'LARGE_OK=YES',
           tiff_path, ecw_path]
    print(f"  [CONV] TIFF → ECW: {os.path.basename(ecw_path)}")
    ok, err = run_gdal(cmd, env, timeout=7200)
    if ok:
        print(f"  [CONV] OK")
        return True
    else:
        print(f"  [CONV] ERRO: {err}")
        return False


# ╔══════════════════════════════════════════════════════════════╗
# ║              FUNÇÕES DE DIAGNÓSTICO                         ║
# ╚══════════════════════════════════════════════════════════════╝


def compression_name(comp):
    return "NONE" if comp is None else str(comp)


def band_statistics(ds, max_sample_pixels=DIAG_MAX_SAMPLE_PIXELS):
    width, height, count = ds.width, ds.height, ds.count
    total_pixels = width * height
    scale = min(1.0, math.sqrt(max_sample_pixels / total_pixels))
    out_w = max(1, int(width * scale))
    out_h = max(1, int(height * scale))
    stats = []
    for band_idx in range(1, count + 1):
        data = ds.read(band_idx, out_shape=(out_h, out_w),
                       resampling=RasterioResampling.nearest).astype("float32")
        nodata = ds.nodata
        mask = (data != nodata) if nodata is not None else np.ones_like(data, dtype=bool)
        valid = data[mask]
        pct_valid = (mask.sum() / data.size) * 100
        if valid.size == 0:
            stats.append({"band": band_idx, "min": None, "max": None, "mean": None,
                          "std": None, "median": None, "pct_valid": 0.0,
                          "is_constant": True, "clipping_low": False, "clipping_high": False})
            continue
        dtype = ds.dtypes[band_idx - 1]
        dtype_max_map = {"uint8": 255, "uint16": 65535, "int16": 32767,
                         "uint32": 4294967295, "int32": 2147483647,
                         "float32": None, "float64": None}
        dtype_max = dtype_max_map.get(dtype)
        vmin, vmax, vmean, vstd = float(valid.min()), float(valid.max()), float(valid.mean()), float(valid.std())
        vmedian = float(np.median(valid))
        clip_low = clip_high = False
        if dtype_max is not None:
            dtype_info = np.iinfo(np.dtype(dtype))
            clip_low = float((valid == dtype_info.min).sum() / valid.size) > 0.005
            clip_high = float((valid == dtype_max).sum() / valid.size) > 0.005
        stats.append({"band": band_idx, "min": round(vmin, 4), "max": round(vmax, 4),
                      "mean": round(vmean, 4), "std": round(vstd, 4), "median": round(vmedian, 4),
                      "pct_valid": round(pct_valid, 2), "is_constant": vmin == vmax,
                      "clipping_low": clip_low, "clipping_high": clip_high})
    return stats, scale


def benchmark_tile_read(ds):
    block_shapes = ds.block_shapes
    if not block_shapes or block_shapes[0][0] is None:
        return None
    tile_h, tile_w = block_shapes[0]
    cx, cy = ds.width // 2, ds.height // 2
    col_off = max(0, cx - tile_w // 2)
    row_off = max(0, cy - tile_h // 2)
    win = Window(col_off, row_off, min(tile_w, ds.width - col_off), min(tile_h, ds.height - row_off))
    t0 = time.perf_counter()
    ds.read(1, window=win)
    return time.perf_counter() - t0


def suggested_overview_levels(width, height, tile_size=256):
    levels, factor, w, h = [], 2, width, height
    while w > tile_size or h > tile_size:
        levels.append(factor)
        w //= 2; h //= 2; factor *= 2
        if factor > 1024: break
    return levels


def extract_metadata_tags(ds):
    meta = {}
    for ns in [None, "EXIF", "XMP", "TIFF", "IMAGE_STRUCTURE"]:
        try:
            tags = ds.tags(ns=ns) if ns else ds.tags()
            if tags: meta[ns or "DEFAULT"] = tags
        except Exception: pass
    return meta


def calc_gsd_and_area(ds):
    crs, transform = ds.crs, ds.transform
    res_x, res_y = abs(transform.a), abs(transform.e)
    is_projected = crs.is_projected if crs else False
    gsd_cm = res_x * 100 if is_projected else None
    area_m2 = res_x * res_y * ds.width * ds.height if is_projected else None
    area_ha = area_m2 / 10000 if area_m2 else None
    area_km2 = area_m2 / 1e6 if area_m2 else None
    return gsd_cm, area_ha, area_km2, is_projected


def estimate_qgis_ram(w, h, count, bpp):
    return w * h * count * bpp * 3


def has_rotation(t):
    return t.b != 0 or t.d != 0


def get_color_interp(ds):
    try:
        return [str(ds.colorinterp[i]) for i in range(ds.count)]
    except Exception: return None


def diagnostic_report(path):
    if not os.path.exists(path):
        return {"erro": "Arquivo não encontrado", "arquivo": path}
    file_size = os.path.getsize(path)
    ext = os.path.splitext(path)[1].lower()
    report = {"arquivo": os.path.abspath(path), "formato": ext,
              "tamanho_disco": file_size, "tamanho_disco_human": human_size(file_size)}
    try:
        with rasterio.open(path) as ds:
            width, height, count = ds.width, ds.height, ds.count
            dtypes, crs, transform = ds.dtypes, ds.crs, ds.transform
            nodata, bounds = ds.nodata, ds.bounds
            res_x, res_y = abs(transform.a), abs(transform.e)
            driver, compression = ds.driver, compression_name(ds.compression)
            tiled, block_shapes = ds.is_tiled, ds.block_shapes
            profile = ds.profile
            predictor = profile.get("predictor") or (
                ds.tags(ns="IMAGE_STRUCTURE").get("PREDICTOR") if ds.tags(ns="IMAGE_STRUCTURE") else None
            )
            photometric = None
            try:
                photometric = ds.tags(ns="TIFF").get("PHOTOMETRIC") or ds.tags().get("PHOTOMETRIC")
            except Exception: pass
            overview_levels = ds.overviews(1) if ds.count >= 1 else []
            num_overviews = len(overview_levels)
            has_alpha = count == 4
            dtype = dtypes[0]
            bpp_map = {"uint8": 1, "int8": 1, "uint16": 2, "int16": 2,
                       "uint32": 4, "int32": 4, "float32": 4, "float64": 8}
            bpp = bpp_map.get(dtype, 4)
            tw, th = block_shapes[0] if block_shapes else (None, None)
            tile_bytes = (tw * th * count * bpp) if tw and th else 0
            raw_size = width * height * count * bpp
            megapixels = (width * height) / 1e6
            is_bigtiff = profile.get("bigtiff", "").upper() in ("YES", "IF_SAFER", "IF_NEEDED")
            gsd_cm, area_ha, area_km2, is_projected = calc_gsd_and_area(ds)
            ram_est = estimate_qgis_ram(width, height, count, bpp)
            rotated = has_rotation(transform)
            meta_tags = extract_metadata_tags(ds)
            suggested_ovrs = suggested_overview_levels(width, height, tw or 256)
            band_stats, _ = band_statistics(ds)
            io_time = benchmark_tile_read(ds)
            ci = get_color_interp(ds)
            problems = []
            if dtype == "float32": problems.append("Raster FLOAT32 → muito pesado")
            if dtype == "float64": problems.append("Raster FLOAT64 → extremamente pesado")
            if compression in ["NONE", "None"]: problems.append("Raster SEM compressao")
            if not tiled and ext != ".ecw": problems.append("Raster nao tiled (strip layout)")
            if count >= 3 and dtype == "uint16": problems.append("RGB em UINT16")
            if file_size > 20 * 1024**3: problems.append("Arquivo >20 GB")
            if num_overviews == 0 and ext != ".ecw": problems.append("Sem overviews internos")
            elif suggested_ovrs and any(o not in overview_levels for o in suggested_ovrs) and ext != ".ecw":
                missing = [o for o in suggested_ovrs if o not in overview_levels]
                problems.append(f"Overviews incompletos: {missing}")
            if predictor is None and any(x in str(compression).lower() for x in ["deflate", "lzw"]):
                problems.append("Compressao sem PREDICTOR")
            if has_alpha: problems.append("4 bandas (alpha)")
            if res_x < 0.01: problems.append("Resolucao extremamente alta")
            if not is_projected: problems.append("CRS nao projetado")
            if rotated: problems.append("Raster com rotacao/skew")
            if not is_bigtiff and raw_size > 4 * 1024**3: problems.append("Arquivo >4GB sem BigTIFF")
            if ram_est > 16 * 1024**3: problems.append(f"RAM estimada QGIS: {human_size(ram_est)}")
            if ext == ".ecw": problems.append("Formato ECW proprietário")
            for s in band_stats:
                if s["is_constant"]: problems.append(f"Banda {s['band']} constante")
                if s["clipping_high"]: problems.append(f"Banda {s['band']} saturacao")
                if s["clipping_low"]: problems.append(f"Banda {s['band']} subexposicao")
                if s["pct_valid"] < 50: problems.append(f"Banda {s['band']}: {s['pct_valid']:.1f}% validos")
            report.update({
                "driver": driver, "largura": width, "altura": height, "bandas": count,
                "megapixels": round(megapixels, 2), "dtype": dtype, "crs": str(crs),
                "is_projected": is_projected, "res_x": res_x, "res_y": res_y,
                "gsd_cm": round(gsd_cm, 4) if gsd_cm else None,
                "area_ha": round(area_ha, 4) if area_ha else None,
                "area_km2": round(area_km2, 6) if area_km2 else None,
                "nodata": nodata, "compressao": compression, "predictor": predictor,
                "tiled": tiled, "block_shapes": [list(b) for b in block_shapes] if block_shapes else [],
                "photometric": photometric, "color_interpretation": ci,
                "overviews": overview_levels, "overviews_sugeridos": suggested_ovrs,
                "bigtiff": is_bigtiff, "rotacao": rotated, "has_alpha": has_alpha,
                "tamanho_raw": raw_size, "taxa_compressao": round(raw_size/file_size, 2) if file_size else 0,
                "ram_estimada_qgis": ram_est, "ram_estimada_qgis_human": human_size(ram_est),
                "benchmark_tile_ms": round(io_time * 1000, 2) if io_time else None,
                "band_stats": band_stats, "meta_tags": {k: dict(v) for k, v in meta_tags.items()},
                "problemas": problems, "diagnostico_ok": True
            })
    except Exception as e:
        report["erro"] = str(e)
        report["diagnostico_ok"] = False
    return report


def save_diagnostic_json(report, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"  [DIAG] Salvo: {output_path}")


# ╔══════════════════════════════════════════════════════════════╗
# ║            FUNÇÃO DE MOSAICO FEATHERING                    ║
# ╚══════════════════════════════════════════════════════════════╝


def ensure_4band_rgb(ds):
    count = ds.count
    dtype = ds.dtypes[0]
    dtype_max = np.iinfo(dtype).max if "int" in dtype else 1.0
    data = ds.read().astype(np.float32)
    if count == 4:
        return data[:3], data[3]
    elif count == 3:
        try:
            mask = ds.dataset_mask().astype(np.float32); mask[mask > 0] = dtype_max
            return data[:3], mask
        except Exception:
            return data[:3], np.ones_like(data[0], dtype=np.float32) * dtype_max
    elif count == 2:
        return np.stack([data[0], data[0], data[0]]), data[1]
    elif count == 1:
        try:
            mask = ds.dataset_mask().astype(np.float32); mask[mask > 0] = dtype_max
            return np.stack([data[0], data[0], data[0]]), mask
        except Exception:
            return np.stack([data[0], data[0], data[0]]), np.ones_like(data[0], dtype=np.float32) * dtype_max
    else:
        return data[:3], data[3] if count > 3 else np.ones_like(data[0], dtype=np.float32) * dtype_max


def run_mosaic(input_files, output_path):
    is_ecw_output = output_path.lower().endswith('.ecw')
    print(f"\n{'='*60}")
    print(f"MOSAICO FEATHERING")
    print(f"  Saída em: {'ECW' if is_ecw_output else 'GeoTIFF'}")
    print(f"{'='*60}")
    print(f"  Arquivos: {len(input_files)}")
    for f in input_files: print(f"    - {os.path.basename(f)}")
    print(f"  Saída: {output_path}")

    # Abrir datasets
    datasets = [rasterio.open(f) for f in input_files]

    # Grid de referência
    print("\n  [MOSAICO] Criando grid de referência...")
    ref_ds = datasets[0]
    out_crs, out_dtype = ref_ds.crs, ref_ds.dtypes[0]
    try:
        mosaic_ref, out_transform = merge(datasets, method='first')
        _, height, width = mosaic_ref.shape
        del mosaic_ref
    except Exception as e:
        print(f"  [ERRO] {e}")
        for ds in datasets: ds.close()
        return None

    print(f"  [MOSAICO] Grid: {width}x{height}  dtype={out_dtype}")
    if "int" in str(out_dtype):
        iinfo = np.iinfo(out_dtype)
        dtype_min, dtype_max = iinfo.min, iinfo.max
    else:
        dtype_min, dtype_max = 0, 1.0

    # Acumuladores
    accum_rgb = np.zeros((3, height, width), dtype=np.float32)
    accum_alpha = np.zeros((height, width), dtype=np.float32)
    accum_w = np.zeros((height, width), dtype=np.float32)

    # Processar cada imagem
    for ds in datasets:
        name = os.path.basename(ds.name)
        print(f"\n  [MOSAICO] {name}...")
        rgb_orig, alpha_orig = ensure_4band_rgb(ds)
        try:
            mask = ds.dataset_mask().astype(np.float32) > 0
        except Exception:
            mask = alpha_orig > 0
        dist = distance_transform_edt(mask).astype(np.float32)
        dmax = dist.max()
        if dmax > 0: dist /= dmax
        del mask
        peso = np.zeros((height, width), dtype=np.float32)
        reproject(source=dist, destination=peso, src_transform=ds.transform, src_crs=ds.crs,
                  dst_transform=out_transform, dst_crs=out_crs, resampling=WarpResampling.bilinear)
        del dist
        dst_rgb = np.zeros((3, height, width), dtype=np.float32)
        reproject(source=rgb_orig, destination=dst_rgb, src_transform=ds.transform, src_crs=ds.crs,
                  dst_transform=out_transform, dst_crs=out_crs, resampling=WarpResampling.bilinear)
        del rgb_orig
        dst_alpha = np.zeros((height, width), dtype=np.float32)
        reproject(source=alpha_orig, destination=dst_alpha, src_transform=ds.transform, src_crs=ds.crs,
                  dst_transform=out_transform, dst_crs=out_crs, resampling=WarpResampling.bilinear)
        del alpha_orig
        if peso.max() == 0:
            peso = dst_alpha / dtype_max
            if peso.max() == 0: peso = np.ones_like(peso, dtype=np.float32) * 0.001
        accum_rgb += dst_rgb * peso
        accum_alpha += dst_alpha * peso
        accum_w += peso
        del dst_rgb, dst_alpha, peso
        print(f"    OK")

    # Normalizar
    print("\n  [MOSAICO] Normalizando...")
    valido = accum_w > 0
    resultado = np.zeros((4, height, width), dtype=out_dtype)
    tmp = np.zeros((height, width), dtype=np.float32)
    for c in range(3):
        np.divide(accum_rgb[c], accum_w, out=tmp, where=valido)
        resultado[c] = np.clip(tmp, dtype_min, dtype_max).astype(out_dtype)
    np.divide(accum_alpha, accum_w, out=tmp, where=valido)
    resultado[3] = np.clip(tmp, dtype_min, dtype_max).astype(out_dtype)
    del accum_rgb, accum_alpha, accum_w, valido, tmp

    # Salvar TIFF intermediário
    tiff_intermediario = output_path
    if is_ecw_output:
        tiff_intermediario = output_path.replace('.ecw', '_temp.tif')

    print(f"\n  [MOSAICO] Salvando TIFF...")
    os.makedirs(os.path.dirname(tiff_intermediario), exist_ok=True)
    with rasterio.open(tiff_intermediario, 'w', driver='GTiff',
                       height=height, width=width, count=4, dtype=out_dtype,
                       crs=out_crs, transform=out_transform,
                       compress=OUTPUT_COMPRESS, predictor=OUTPUT_PREDICTOR,
                       tiled=OUTPUT_TILED, blockxsize=OUTPUT_BLOCK_X, blockysize=OUTPUT_BLOCK_Y,
                       interleave=OUTPUT_INTERLEAVE, nodata=OUTPUT_NODATA) as dst:
        dst.write(resultado)
        dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha)
    del resultado

    # Overviews
    print("  [MOSAICO] Overviews...")
    with rasterio.open(tiff_intermediario, 'r+') as dst:
        dst.build_overviews(OVERVIEW_LEVELS, OVERVIEW_RESAMPLING)
        dst.update_tags(ns='rio_overview', resampling='nearest')

    # Converter para ECW se necessário
    if is_ecw_output:
        gdal_cmd, env = setup_gdal_environment()
        if gdal_cmd:
            if convert_tiff_to_ecw(tiff_intermediario, output_path, gdal_cmd, env):
                if not KEEP_TEMP_TIFFS:
                    os.remove(tiff_intermediario)
                    print(f"  [MOSAICO] Temp removido")
                print(f"\n  [MOSAICO] ECW concluído!")
            else:
                print(f"\n  [MOSAICO] Falha ECW. Mantendo TIFF.")
                fallback = output_path.replace('.ecw', '_fallback.tif')
                shutil.move(tiff_intermediario, fallback)
                output_path = fallback
        else:
            print(f"\n  [MOSAICO] Plugin ECW não disponível. Mantendo TIFF.")
            fallback = output_path.replace('.ecw', '_fallback.tif')
            shutil.move(tiff_intermediario, fallback)
            output_path = fallback

    for ds in datasets: ds.close()
    print(f"\n  [MOSAICO] Concluído: {output_path}")
    return output_path


# ╔══════════════════════════════════════════════════════════════╗
# ║                      FUNÇÃO PRINCIPAL                       ║
# ╚══════════════════════════════════════════════════════════════╝


def main():
    print("=" * 70)
    print(" MOSAICO MERGE + DIAGNÓSTICO (ECW)")
    print("=" * 70)
    print(f"\n  INPUT_DIR  : {INPUT_DIR}")
    print(f"  OUTPUT_DIR : {OUTPUT_DIR}")
    print(f"  OUTPUT_FILE: {OUTPUT_FILENAME}")
    print(f"  FORMATOS   : {', '.join(INPUT_EXTENSIONS)}")

    # Configurar GDAL com plugin ECW
    gdal_cmd, gdal_env = setup_gdal_environment()
    if gdal_cmd:
        print(f"  ✓ Plugin ECW configurado!")
    else:
        print(f"  ⚠ Plugin ECW não encontrado no QGIS. Tente: copiar gdal_ECW_JP2ECW.dll para pasta do script")
        print(f"    Procure em: C:\\Program Files\\QGIS 3.16\\bin\\gdalplugins\\gdal_ECW_JP2ECW.dll")

    # Criar output dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Encontrar arquivos
    input_files = []
    for pattern in INPUT_EXTENSIONS:
        input_files.extend(glob.glob(os.path.join(INPUT_DIR, pattern)))
    input_files = sorted(set(input_files))
    input_files = [f for f in input_files if os.path.basename(OUTPUT_FILENAME) not in f]

    if len(input_files) == 0:
        print(f"\n  ERRO: Nenhum arquivo em {INPUT_DIR}")
        print(f"  Procurando: {', '.join(INPUT_EXTENSIONS)}")
        sys.exit(1)

    print(f"\n  Arquivos: {len(input_files)}")
    for i, f in enumerate(input_files, 1):
        print(f"    [{i}] ({os.path.splitext(f)[1].upper():>5}) {human_size(os.path.getsize(f)):>8}  {os.path.basename(f)}")

    # Converter ECWs para TIFF
    arquivos_processados = []
    temp_dir = tempfile.mkdtemp(prefix="mosaico_ecw_")

    for f in input_files:
        if f.lower().endswith('.ecw'):
            if gdal_cmd:
                tiff_temp = os.path.join(temp_dir, os.path.basename(f).replace('.ecw', '.tif'))
                print(f"\n  [PRE] Convertendo {os.path.basename(f)}...")
                if convert_ecw_to_tiff(f, tiff_temp, gdal_cmd, gdal_env):
                    arquivos_processados.append(tiff_temp)
                else:
                    print(f"  [PRE] Pulando {os.path.basename(f)}...")
            else:
                print(f"\n  [PRE] ERRO: Não pode ler {os.path.basename(f)} (sem plugin ECW)")
        else:
            arquivos_processados.append(f)

    if len(arquivos_processados) == 0:
        print(f"\n  ERRO: Nenhum arquivo processado.")
        shutil.rmtree(temp_dir, ignore_errors=True)
        sys.exit(1)

    print(f"\n  Prontos: {len(arquivos_processados)}")

    # Diagnóstico dos inputs
    if RUN_DIAGNOSTIC_INPUTS:
        print(f"\n{'='*70}\n DIAGNÓSTICO ENTRADA\n{'='*70}")
        for f in input_files:
            base = os.path.splitext(os.path.basename(f))[0]
            json_path = os.path.join(OUTPUT_DIR, f"diagnostico_input_{base}.json")
            print(f"\n  {os.path.basename(f)}")
            save_diagnostic_json(diagnostic_report(f), json_path)

    # Mosaico
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    if RUN_MOSAIC:
        if os.path.exists(output_path):
            print(f"\n  ATENÇÃO: '{output_path}' existe, sobrescrevendo.")
        resultado_path = run_mosaic(arquivos_processados, output_path)
        if resultado_path:
            output_path = resultado_path
        else:
            print(f"\n  ERRO no mosaico")
            shutil.rmtree(temp_dir, ignore_errors=True)
            sys.exit(1)
    else:
        if not os.path.exists(output_path):
            print(f"\n  AVISO: mosaico desabilitado e saída não existe")
            shutil.rmtree(temp_dir, ignore_errors=True)
            sys.exit(1)

    # Diagnóstico output
    if RUN_DIAGNOSTIC_OUTPUT and os.path.exists(output_path):
        print(f"\n{'='*70}\n DIAGNÓSTICO SAÍDA\n{'='*70}")
        json_out = os.path.join(OUTPUT_DIR, "diagnostico_output.json")
        print(f"\n  {os.path.basename(output_path)}")
        report = diagnostic_report(output_path)
        save_diagnostic_json(report, json_out)

        print(f"\n  {'='*50}")
        print(f"  RESUMO")
        print(f"  {'='*50}")
        print(f"  Arquivo : {output_path}")
        print(f"  Tamanho : {human_size(report.get('tamanho_disco', 0))}")
        print(f"  Dimensão: {report.get('largura', '?')}x{report.get('altura', '?')}")
        print(f"  Bandas  : {report.get('bandas', '?')}")
        if report.get('gsd_cm'): print(f"  GSD     : {report['gsd_cm']:.2f} cm/px")
        if report.get('area_ha'): print(f"  Área    : {report['area_ha']:.2f} ha")
        problemas = report.get('problemas', [])
        if problemas:
            print(f"  Problemas: {len(problemas)}")
            for p in problemas[:5]: print(f"    - {p}")
            if len(problemas) > 5: print(f"    ... +{len(problemas)-5}")
        else: print(f"  Problemas: Nenhum ✓")
        print(f"  {'='*50}")

    # Listar output
    print(f"\n{'='*70}\n ARQUIVOS GERADOS\n{'='*70}")
    for f in sorted(glob.glob(os.path.join(OUTPUT_DIR, "*"))):
        print(f"  {human_size(os.path.getsize(f)):>8}  {os.path.basename(f)}")

    # Limpar
    if not KEEP_TEMP_TIFFS:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"\n{'='*70}")
    print(" CONCLUÍDO!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()