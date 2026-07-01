# -*- coding: utf-8 -*-
"""
MOSAICO MERGE + DIAGNÓSTICO (ECW) v3
=====================================
v3: 
  - Se apenas 1 arquivo: apenas converte formato (ECW <-> TIFF)
  - Se 2+ arquivos: mosaico feathering completo
  - Saída definida pela extensão do OUTPUT_FILENAME (.ecw ou .tif)
  - Processamento em blocos (tiles) para evitar erro de memória
  - Timestamp [HH:MM:SS] em todas as mensagens
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
from datetime import datetime
import numpy as np
from scipy.ndimage import distance_transform_edt
import rasterio
from rasterio.warp import reproject, Resampling as WarpResampling
from rasterio.merge import merge
from rasterio.enums import ColorInterp, Resampling as RasterioResampling
from rasterio.windows import Window

warnings.filterwarnings("ignore")


def ts():
    return datetime.now().strftime("[%H:%M:%S]")


def log(msg):
    print(f"{ts()} {msg}")


# ╔══════════════════════════════════════════════════════════════╗
# ║         CONFIGURAÇÕES                                       ║
# ╚══════════════════════════════════════════════════════════════╝

INPUT_DIR  = r"G:\BR070\ETAPA2\OUTPUT\ORTOMOSAICO_FINAL_01\e2"
OUTPUT_DIR = r"G:\BR070\ETAPA2\OUTPUT\ORTOMOSAICO_FINAL_01\e2"
OUTPUT_FILENAME = "mosaico_final.ecw"
INPUT_EXTENSIONS = ["*.tif", "*.tiff", "*.ecw"]

OUTPUT_COMPRESS   = "deflate"
OUTPUT_PREDICTOR  = 2
OUTPUT_TILED      = True
OUTPUT_BLOCK_X    = 512
OUTPUT_BLOCK_Y    = 512
OUTPUT_INTERLEAVE = "pixel"
OUTPUT_NODATA     = None

ECW_TARGET_PERCENT = 90

OVERVIEW_LEVELS = [2, 4, 8, 16, 32, 64, 128, 256]
OVERVIEW_RESAMPLING = WarpResampling.nearest
DIAG_MAX_SAMPLE_PIXELS = 10_000_000

RUN_DIAGNOSTIC_INPUTS  = True
RUN_DIAGNOSTIC_OUTPUT  = True
RUN_MOSAIC             = True
KEEP_TEMP_TIFFS        = False

# ── Processamento em blocos (tiles) para evitar falta de RAM ──
#   O merge completo de rasters gigantes pode exigir >500 GB RAM.
#   Processamos em blocos de TILE_SIZE x TILE_SIZE pixels.
TILE_SIZE = 4096  # pixels por bloco (ajuste conforme RAM disponivel)

# ╔══════════════════════════════════════════════════════════════╗
# ║       CONFIGURAÇÃO GDAL — QGIS + PLUGIN ECW                ║
# ╚══════════════════════════════════════════════════════════════╝

QGIS_VERSIONS = [
    {"path": r"C:\Program Files\QGIS 3.16", "gdal_plugins": r"bin\gdalplugins", "bin": r"bin\gdal_translate.exe"},
    {"path": r"C:\Program Files\QGIS 3.34.12", "gdal_plugins": r"apps\gdal\lib\gdalplugins", "bin": r"bin\gdal_translate.exe"},
    {"path": r"C:\Program Files\QGIS 3.40.14", "gdal_plugins": r"apps\gdal\lib\gdalplugins", "bin": r"bin\gdal_translate.exe"},
    {"path": r"C:\Program Files\QGIS 4.0.0", "gdal_plugins": r"apps\gdal\lib\gdalplugins", "bin": r"bin\gdal_translate.exe"},
]


def setup_gdal_environment():
    for qver in QGIS_VERSIONS:
        gdal_bin = os.path.join(qver["path"], qver["bin"])
        plugin_dir = os.path.join(qver["path"], qver["gdal_plugins"])
        ecw_plugin = os.path.join(plugin_dir, "gdal_ECW_JP2ECW.dll")
        if os.path.exists(gdal_bin) and os.path.exists(plugin_dir):
            has_ecw = os.path.exists(ecw_plugin) or any("ECW" in f for f in os.listdir(plugin_dir))
            if has_ecw:
                env = os.environ.copy()
                env["GDAL_DRIVER_PATH"] = plugin_dir
                try:
                    result = subprocess.run([gdal_bin, "--version"], capture_output=True, text=True, timeout=10, env=env)
                    if result.returncode == 0:
                        log(f"GDAL: {qver['path']}")
                        log(f"Plugin ECW: {plugin_dir}")
                        return gdal_bin, env
                except Exception: pass
    return None, None


# ╔══════════════════════════════════════════════════════════════╗
# ║              FUNÇÕES AUXILIARES                             ║
# ╚══════════════════════════════════════════════════════════════╝


def human_size(size):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024: return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


def run_gdal_monitor(cmd, env, timeout=86400):
    """Executa GDAL com monitoramento a cada 60s."""
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        t_start = time.time()
        last_feedback = time.time()
        while process.poll() is None:
            time.sleep(5)
            elapsed = time.time() - t_start
            if time.time() - last_feedback >= 60:
                mins, secs = int(elapsed // 60), int(elapsed % 60)
                log(f"[CONV] Processando... ({mins:02d}:{secs:02d})")
                last_feedback = time.time()
            if elapsed > timeout:
                process.kill()
                return False, f"Timeout {elapsed:.0f}s"
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            return False, stderr[:500] if stderr else f"Erro {process.returncode}"
        return True, ""
    except Exception as e:
        return False, str(e)


def convert_ecw_to_tiff(ecw_path, tiff_path, gdal_cmd, env):
    cmd = [gdal_cmd, '-of', 'GTiff', '-co', 'COMPRESS=DEFLATE', '-co', 'PREDICTOR=2',
           '-co', 'TILED=YES', '-co', 'BIGTIFF=IF_SAFER', ecw_path, tiff_path]
    log(f"[CONV] ECW -> TIFF: {os.path.basename(ecw_path)}")
    t_start = time.time()
    ok, err = run_gdal_monitor(cmd, env, timeout=86400)
    t_elapsed = time.time() - t_start
    if ok:
        size_gb = os.path.getsize(tiff_path) / (1024**3)
        log(f"[CONV] OK! ({t_elapsed:.0f}s) {size_gb:.1f} GB")
        return True
    else:
        log(f"[CONV] ERRO ({t_elapsed:.0f}s): {err}")
        return False


def convert_tiff_to_ecw(tiff_path, ecw_path, gdal_cmd, env):
    cmd = [gdal_cmd, '-of', 'ECW', '-co', f'TARGET={ECW_TARGET_PERCENT}', '-co', 'LARGE_OK=YES', tiff_path, ecw_path]
    log(f"[CONV] TIFF -> ECW: {os.path.basename(ecw_path)}")
    t_start = time.time()
    ok, err = run_gdal_monitor(cmd, env, timeout=86400)
    t_elapsed = time.time() - t_start
    if ok:
        size_mb = os.path.getsize(ecw_path) / (1024 * 1024)
        log(f"[CONV] OK! ({t_elapsed:.0f}s) {size_mb:.1f} MB")
        return True
    else:
        log(f"[CONV] ERRO ({t_elapsed:.0f}s): {err}")
        return False


def convert_format(input_path, output_path, gdal_cmd, env):
    """
    Converte formato de um arquivo raster usando gdal_translate.
    Detecta automaticamente o formato de entrada/saída pela extensão.
    """
    ext_in = os.path.splitext(input_path)[1].lower()
    ext_out = os.path.splitext(output_path)[1].lower()
    
    log(f"[CONV] {ext_in.upper()} -> {ext_out.upper()}: {os.path.basename(input_path)}")
    
    # Determinar formato de saída
    of_map = {'.tif': 'GTiff', '.tiff': 'GTiff', '.ecw': 'ECW'}
    of = of_map.get(ext_out, 'GTiff')
    
    # Parâmetros específicos por formato
    co_params = []
    if of == 'GTiff':
        co_params = ['-co', 'COMPRESS=DEFLATE', '-co', 'PREDICTOR=2', '-co', 'TILED=YES', '-co', 'BIGTIFF=IF_SAFER']
    elif of == 'ECW':
        co_params = ['-co', f'TARGET={ECW_TARGET_PERCENT}', '-co', 'LARGE_OK=YES']
    
    cmd = [gdal_cmd, '-of', of] + co_params + [input_path, output_path]
    
    t_start = time.time()
    ok, err = run_gdal_monitor(cmd, env, timeout=86400)
    t_elapsed = time.time() - t_start
    if ok:
        size = os.path.getsize(output_path)
        log(f"[CONV] OK! ({t_elapsed:.0f}s) {human_size(size)}")
        return True
    else:
        log(f"[CONV] ERRO ({t_elapsed:.0f}s): {err}")
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
        data = ds.read(band_idx, out_shape=(out_h, out_w), resampling=RasterioResampling.nearest).astype("float32")
        nodata = ds.nodata
        mask = (data != nodata) if nodata is not None else np.ones_like(data, dtype=bool)
        valid = data[mask]
        pct_valid = (mask.sum() / data.size) * 100
        if valid.size == 0:
            stats.append({"band": band_idx, "min": None, "max": None, "mean": None, "std": None, "median": None, "pct_valid": 0.0, "is_constant": True, "clipping_low": False, "clipping_high": False})
            continue
        dtype = ds.dtypes[band_idx - 1]
        dtype_max_map = {"uint8": 255, "uint16": 65535, "int16": 32767, "uint32": 4294967295, "int32": 2147483647, "float32": None, "float64": None}
        dtype_max = dtype_max_map.get(dtype)
        vmin, vmax, vmean, vstd = float(valid.min()), float(valid.max()), float(valid.mean()), float(valid.std())
        vmedian = float(np.median(valid))
        clip_low = clip_high = False
        if dtype_max is not None:
            dtype_info = np.iinfo(np.dtype(dtype))
            clip_low = float((valid == dtype_info.min).sum() / valid.size) > 0.005
            clip_high = float((valid == dtype_max).sum() / valid.size) > 0.005
        stats.append({"band": band_idx, "min": round(vmin, 4), "max": round(vmax, 4), "mean": round(vmean, 4), "std": round(vstd, 4), "median": round(vmedian, 4), "pct_valid": round(pct_valid, 2), "is_constant": vmin == vmax, "clipping_low": clip_low, "clipping_high": clip_high})
    return stats, scale


def benchmark_tile_read(ds):
    block_shapes = ds.block_shapes
    if not block_shapes or block_shapes[0][0] is None: return None
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
    try: return [str(ds.colorinterp[i]) for i in range(ds.count)]
    except Exception: return None


def diagnostic_report(path):
    if not os.path.exists(path):
        return {"erro": "Arquivo nao encontrado", "arquivo": path}
    file_size = os.path.getsize(path)
    ext = os.path.splitext(path)[1].lower()
    report = {"arquivo": os.path.abspath(path), "formato": ext, "tamanho_disco": file_size, "tamanho_disco_human": human_size(file_size)}
    try:
        with rasterio.open(path) as ds:
            width, height, count = ds.width, ds.height, ds.count
            dtypes, crs, transform = ds.dtypes, ds.crs, ds.transform
            nodata, bounds = ds.nodata, ds.bounds
            res_x, res_y = abs(transform.a), abs(transform.e)
            driver, compression = ds.driver, compression_name(ds.compression)
            tiled, block_shapes = ds.is_tiled, ds.block_shapes
            profile = ds.profile
            predictor = profile.get("predictor") or (ds.tags(ns="IMAGE_STRUCTURE").get("PREDICTOR") if ds.tags(ns="IMAGE_STRUCTURE") else None)
            photometric = None
            try: photometric = ds.tags(ns="TIFF").get("PHOTOMETRIC") or ds.tags().get("PHOTOMETRIC")
            except Exception: pass
            overview_levels = ds.overviews(1) if ds.count >= 1 else []
            num_overviews = len(overview_levels)
            has_alpha = count == 4
            dtype = dtypes[0]
            bpp_map = {"uint8": 1, "int8": 1, "uint16": 2, "int16": 2, "uint32": 4, "int32": 4, "float32": 4, "float64": 8}
            bpp = bpp_map.get(dtype, 4)
            tw, th = block_shapes[0] if block_shapes else (None, None)
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
            if dtype == "float32": problems.append("Raster FLOAT32 -> muito pesado")
            if dtype == "float64": problems.append("Raster FLOAT64 -> extremamente pesado")
            if compression in ["NONE", "None"]: problems.append("Raster SEM compressao")
            if not tiled and ext != ".ecw": problems.append("Raster nao tiled (strip layout)")
            if count >= 3 and dtype == "uint16": problems.append("RGB em UINT16")
            if file_size > 20 * 1024**3: problems.append("Arquivo >20 GB")
            if num_overviews == 0 and ext != ".ecw": problems.append("Sem overviews internos")
            elif suggested_ovrs and any(o not in overview_levels for o in suggested_ovrs) and ext != ".ecw":
                missing = [o for o in suggested_ovrs if o not in overview_levels]
                problems.append(f"Overviews incompletos: {missing}")
            if predictor is None and any(x in str(compression).lower() for x in ["deflate", "lzw"]): problems.append("Compressao sem PREDICTOR")
            if has_alpha: problems.append("4 bandas (alpha)")
            if res_x < 0.01: problems.append("Resolucao extremamente alta")
            if not is_projected: problems.append("CRS nao projetado")
            if rotated: problems.append("Raster com rotacao/skew")
            if not is_bigtiff and raw_size > 4 * 1024**3: problems.append("Arquivo >4GB sem BigTIFF")
            if ram_est > 16 * 1024**3: problems.append(f"RAM estimada QGIS: {human_size(ram_est)}")
            if ext == ".ecw": problems.append("Formato ECW proprietario")
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
    log(f"[DIAG] Salvo: {output_path}")


# ╔══════════════════════════════════════════════════════════════╗
# ║            MOSAICO EM BLOCOS (TILED)                       ║
# ╚══════════════════════════════════════════════════════════════╝


def ensure_4band_rgb(ds):
    count = ds.count
    dtype = ds.dtypes[0]
    dtype_max = np.iinfo(dtype).max if "int" in dtype else 1.0
    data = ds.read().astype(np.float32)
    if count == 4: return data[:3], data[3]
    elif count == 3:
        try:
            mask = ds.dataset_mask().astype(np.float32); mask[mask > 0] = dtype_max
            return data[:3], mask
        except Exception: return data[:3], np.ones_like(data[0], dtype=np.float32) * dtype_max
    elif count == 2: return np.stack([data[0], data[0], data[0]]), data[1]
    elif count == 1:
        rgb = np.stack([data[0], data[0], data[0]])
        try:
            mask = ds.dataset_mask().astype(np.float32); mask[mask > 0] = dtype_max
            return rgb, mask
        except Exception: return rgb, np.ones_like(data[0], dtype=np.float32) * dtype_max
    else: return data[:3], data[3] if count > 3 else np.ones_like(data[0], dtype=np.float32) * dtype_max


def run_mosaic_tiled(input_files, output_path):
    """
    Mosaico feathering processado em blocos (tiles) para economizar RAM.
    Em vez de carregar tudo na memória, processa bloco por bloco.
    """
    is_ecw_output = output_path.lower().endswith('.ecw')
    
    log("=" * 60)
    log(f"MOSAICO FEATHERING (TILED - {TILE_SIZE}px blocos)")
    log(f"Saida: {'ECW' if is_ecw_output else 'GeoTIFF'}")
    log("=" * 60)
    log(f"Arquivos: {len(input_files)}")
    for f in input_files: log(f"  - {os.path.basename(f)}")
    log(f"Saida: {output_path}")

    # Abrir datasets
    log("[MOSAICO] Abrindo datasets...")
    datasets = [rasterio.open(f) for f in input_files]
    ref_ds = datasets[0]
    out_crs, out_dtype = ref_ds.crs, ref_ds.dtypes[0]

    # Calcular grid total sem carregar tudo na RAM
    log("[MOSAICO] Calculando grid total...")
    try:
        # Merge com method='first' mas apenas para 1 banda (mais leve)
        # Usamos window para reduzir ainda mais se necessário
        mosaic_ref, out_transform = merge(datasets, method='first')
        _, height, width = mosaic_ref.shape
        del mosaic_ref
    except MemoryError:
        log("[MOSAICO] ERRO: Memoria insuficiente para calcular grid total!")
        log("[MOSAICO] Os rasters sao muito grandes para processar em RAM.")
        log("[MOSAICO] Alternativa: converta cada ECW para TIFF separadamente")
        log("[MOSAICO] e depois use gdal_merge.py ou QGIS para unir.")
        for ds in datasets: ds.close()
        return None
    except Exception as e:
        log(f"[MOSAICO] ERRO no merge: {e}")
        for ds in datasets: ds.close()
        return None

    log(f"[MOSAICO] Grid: {width}x{height} dtype={out_dtype}")
    log(f"[MOSAICO] Megapixels: {(width*height)/1e6:.0f} MP")
    
    # Estimar RAM necessária
    bpp = {"uint8": 1, "uint16": 2, "float32": 4}.get(out_dtype, 1)
    ram_gb = width * height * 4 * bpp * 3 / (1024**3)
    log(f"[MOSAICO] RAM estimada (full): {ram_gb:.0f} GB")
    log(f"[MOSAICO] Usando blocos de {TILE_SIZE}x{TILE_SIZE} para reduzir RAM")

    if "int" in str(out_dtype):
        iinfo = np.iinfo(out_dtype)
        dtype_min, dtype_max = iinfo.min, iinfo.max
    else:
        dtype_min, dtype_max = 0, 1.0

    # Salvar TIFF intermediário
    tiff_intermediario = output_path
    if is_ecw_output:
        tiff_intermediario = output_path.replace('.ecw', '_temp.tif')

    log(f"[MOSAICO] Criando arquivo TIFF de saida...")
    os.makedirs(os.path.dirname(tiff_intermediario), exist_ok=True)
    
    # Criar o arquivo de saída vazio
    with rasterio.open(tiff_intermediario, 'w', driver='GTiff',
                       height=height, width=width, count=4, dtype=out_dtype,
                       crs=out_crs, transform=out_transform,
                       compress=OUTPUT_COMPRESS, predictor=OUTPUT_PREDICTOR,
                       tiled=OUTPUT_TILED, blockxsize=OUTPUT_BLOCK_X, blockysize=OUTPUT_BLOCK_Y,
                       interleave=OUTPUT_INTERLEAVE, nodata=OUTPUT_NODATA) as dst:
        # Escrever um bloco vazio para inicializar
        dst.write(np.zeros((4, height, width), dtype=out_dtype))
    
    log(f"[MOSAICO] Arquivo criado. Processando por blocos...")
    
    # Processar por blocos
    n_tiles_x = math.ceil(width / TILE_SIZE)
    n_tiles_y = math.ceil(height / TILE_SIZE)
    total_tiles = n_tiles_x * n_tiles_y
    tile_count = 0
    
    t_mosaic_start = time.time()
    
    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            tile_count += 1
            col_off = tx * TILE_SIZE
            row_off = ty * TILE_SIZE
            tw = min(TILE_SIZE, width - col_off)
            th = min(TILE_SIZE, height - row_off)
            
            win = Window(col_off, row_off, tw, th)
            
            log(f"[MOSAICO] Bloco {tile_count}/{total_tiles} ({tx+1},{ty+1}) - {tw}x{th} em ({col_off},{row_off})")
            t_tile = time.time()
            
            # Acumuladores para este bloco
            accum_rgb = np.zeros((3, th, tw), dtype=np.float32)
            accum_alpha = np.zeros((th, tw), dtype=np.float32)
            accum_w = np.zeros((th, tw), dtype=np.float32)
            
            # Processar cada dataset para este bloco
            for ds_idx, ds in enumerate(datasets):
                name = os.path.basename(ds.name)
                
                # Ler dados do dataset na região do bloco (reprojetado)
                try:
                    # Reprojetar a região do bloco
                    # Primeiro, ler os dados de origem
                    src_data = ds.read().astype(np.float32)
                    
                    # Criar array de destino para este bloco
                    dst_rgb = np.zeros((3, th, tw), dtype=np.float32)
                    dst_alpha = np.zeros((th, tw), dtype=np.float32)
                    
                    # Reprojetar RGB
                    reproject(
                        source=src_data[:3] if ds.count >= 3 else src_data,
                        destination=dst_rgb,
                        src_transform=ds.transform,
                        src_crs=ds.crs,
                        dst_transform=out_transform,
                        dst_crs=out_crs,
                        resampling=WarpResampling.bilinear,
                        dst_window=win
                    )
                    
                    # Reprojetar alpha
                    if ds.count >= 4:
                        reproject(
                            source=src_data[3:4],
                            destination=dst_alpha,
                            src_transform=ds.transform,
                            src_crs=ds.crs,
                            dst_transform=out_transform,
                            dst_crs=out_crs,
                            resampling=WarpResampling.bilinear,
                            dst_window=win
                        )
                    else:
                        # Criar alpha a partir da máscara
                        try:
                            mask = ds.dataset_mask().astype(np.float32)
                            reproject(
                                source=mask,
                                destination=dst_alpha,
                                src_transform=ds.transform,
                                src_crs=ds.crs,
                                dst_transform=out_transform,
                                dst_crs=out_crs,
                                resampling=WarpResampling.bilinear,
                                dst_window=win
                            )
                        except Exception:
                            dst_alpha.fill(dtype_max)
                    
                    # Peso baseado em distance transform
                    mask = (dst_alpha > 0).astype(np.float32)
                    dist = distance_transform_edt(mask).astype(np.float32)
                    dmax = dist.max()
                    if dmax > 0: dist /= dmax
                    
                    peso = dist
                    if peso.max() == 0:
                        peso = dst_alpha / dtype_max
                        if peso.max() == 0:
                            peso = np.ones_like(peso, dtype=np.float32) * 0.001
                    
                    accum_rgb += dst_rgb * peso
                    accum_alpha += dst_alpha * peso
                    accum_w += peso
                    
                except Exception as e:
                    log(f"[MOSAICO]   Erro processando {name} no bloco: {e}")
            
            # Normalizar e escrever bloco
            valido = accum_w > 0
            resultado_bloco = np.zeros((4, th, tw), dtype=out_dtype)
            tmp = np.zeros((th, tw), dtype=np.float32)
            
            for c in range(3):
                np.divide(accum_rgb[c], accum_w, out=tmp, where=valido)
                resultado_bloco[c] = np.clip(tmp, dtype_min, dtype_max).astype(out_dtype)
            
            np.divide(accum_alpha, accum_w, out=tmp, where=valido)
            resultado_bloco[3] = np.clip(tmp, dtype_min, dtype_max).astype(out_dtype)
            
            # Escrever bloco no arquivo
            with rasterio.open(tiff_intermediario, 'r+') as dst:
                dst.write(resultado_bloco, window=win)
            
            del accum_rgb, accum_alpha, accum_w, valido, tmp, resultado_bloco
            
            elapsed = time.time() - t_tile
            total_elapsed = time.time() - t_mosaic_start
            log(f"[MOSAICO] Bloco OK ({elapsed:.1f}s) - Total: {total_elapsed/60:.1f}min")
    
    # Fechar datasets
    for ds in datasets: ds.close()
    
    # Overviews
    log(f"[MOSAICO] Gerando overviews...")
    t0_ovr = time.time()
    with rasterio.open(tiff_intermediario, 'r+') as dst:
        dst.build_overviews(OVERVIEW_LEVELS, OVERVIEW_RESAMPLING)
        dst.update_tags(ns='rio_overview', resampling='nearest')
    log(f"[MOSAICO] Overviews OK ({time.time()-t0_ovr:.1f}s)")

    # Converter para ECW se necessario
    if is_ecw_output:
        gdal_cmd, env = setup_gdal_environment()
        if gdal_cmd:
            tiff_size = os.path.getsize(tiff_intermediario) / (1024**3)
            log(f"[MOSAICO] Convertendo TIFF ({tiff_size:.1f} GB) -> ECW...")
            if convert_tiff_to_ecw(tiff_intermediario, output_path, gdal_cmd, env):
                if not KEEP_TEMP_TIFFS:
                    os.remove(tiff_intermediario)
                    log(f"[MOSAICO] Temp removido")
                log(f"[MOSAICO] ECW concluido!")
            else:
                log(f"[MOSAICO] Falha ECW. Mantendo TIFF.")
                fallback = output_path.replace('.ecw', '_fallback.tif')
                shutil.move(tiff_intermediario, fallback)
                output_path = fallback
        else:
            log(f"[MOSAICO] Plugin ECW indisponivel. Mantendo TIFF.")
            fallback = output_path.replace('.ecw', '_fallback.tif')
            shutil.move(tiff_intermediario, fallback)
            output_path = fallback

    log(f"[MOSAICO] Concluido: {output_path}")
    return output_path


# ╔══════════════════════════════════════════════════════════════╗
# ║                      FUNÇÃO PRINCIPAL                       ║
# ╚══════════════════════════════════════════════════════════════╝


def main():
    t_inicio = time.time()
    
    log("=" * 70)
    log(" MOSAICO MERGE + DIAGNOSTICO (ECW) v3")
    log("=" * 70)
    log(f"")
    log(f"INPUT_DIR  : {INPUT_DIR}")
    log(f"OUTPUT_DIR : {OUTPUT_DIR}")
    log(f"OUTPUT_FILE: {OUTPUT_FILENAME}")
    log(f"FORMATOS   : {', '.join(INPUT_EXTENSIONS)}")

    # Detectar formato de saída
    output_ext = os.path.splitext(OUTPUT_FILENAME)[1].lower()
    log(f"Formato de saida: {output_ext.upper()}")

    # Configurar GDAL
    gdal_cmd, gdal_env = setup_gdal_environment()
    if gdal_cmd:
        log("Plugin ECW OK!")
    else:
        log("ATENCAO: Plugin ECW nao encontrado")

    # Criar output dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Encontrar arquivos
    input_files = []
    for pattern in INPUT_EXTENSIONS:
        input_files.extend(glob.glob(os.path.join(INPUT_DIR, pattern)))
    input_files = sorted(set(input_files))
    input_files = [f for f in input_files if os.path.basename(OUTPUT_FILENAME) not in f]

    if len(input_files) == 0:
        log(f"ERRO: Nenhum arquivo em {INPUT_DIR}")
        sys.exit(1)

    log(f"")
    log(f"Arquivos: {len(input_files)}")
    for i, f in enumerate(input_files, 1):
        log(f"  [{i}] ({os.path.splitext(f)[1].upper():>5}) {human_size(os.path.getsize(f)):>8}  {os.path.basename(f)}")

    # ── SE FOR APENAS 1 ARQUIVO: CONVERTER FORMATO ─────────────────────
    if len(input_files) == 1:
        log(f"")
        log("=" * 60)
        log(" MODO CONVERSAO: 1 arquivo - apenas convertendo formato")
        log("=" * 60)
        
        input_file = input_files[0]
        input_ext = os.path.splitext(input_file)[1].lower()
        output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
        
        # Se entrada e saída são o mesmo formato, avisar
        if input_ext == output_ext:
            log(f"AVISO: Entrada e saida sao o mesmo formato ({input_ext}).")
            log(f"Copiando arquivo...")
            shutil.copy2(input_file, output_path)
            log(f"Copiado: {output_path}")
        else:
            # Precisa converter
            if gdal_cmd:
                log(f"Convertendo {input_ext.upper()} -> {output_ext.upper()}...")
                
                # Se entrada é ECW, primeiro converte para TIFF temporário
                if input_ext == '.ecw':
                    temp_tiff = os.path.join(tempfile.gettempdir(), f"temp_conv_{int(time.time())}.tif")
                    if convert_ecw_to_tiff(input_file, temp_tiff, gdal_cmd, gdal_env):
                        if output_ext == '.tif':
                            # Quer TIFF como saída
                            shutil.move(temp_tiff, output_path)
                            log(f"Resultado: {output_path}")
                        elif output_ext == '.ecw':
                            # Quer ECW como saída (converter TIFF -> ECW)
                            if convert_tiff_to_ecw(temp_tiff, output_path, gdal_cmd, gdal_env):
                                os.remove(temp_tiff)
                                log(f"Resultado: {output_path}")
                            else:
                                log(f"Falha na conversao para ECW. Mantendo TIFF.")
                                shutil.move(temp_tiff, output_path.replace('.ecw', '.tif'))
                    else:
                        log(f"ERRO: Falha na conversao ECW -> TIFF")
                elif input_ext in ('.tif', '.tiff') and output_ext == '.ecw':
                    # TIFF -> ECW direto
                    if not convert_tiff_to_ecw(input_file, output_path, gdal_cmd, gdal_env):
                        log(f"ERRO: Falha na conversao TIFF -> ECW")
                else:
                    # Outros formatos: usar gdal_translate genérico
                    if not convert_format(input_file, output_path, gdal_cmd, gdal_env):
                        log(f"ERRO: Falha na conversao")
            else:
                log(f"ERRO: gdal_translate nao disponivel para conversao")
                sys.exit(1)
        
        # Diagnóstico do resultado
        if RUN_DIAGNOSTIC_OUTPUT and os.path.exists(output_path):
            log(f"")
            log("=" * 70)
            log(" DIAGNOSTICO DO ARQUIVO DE SAIDA")
            log("=" * 70)
            json_out = os.path.join(OUTPUT_DIR, "diagnostico_output.json")
            report = diagnostic_report(output_path)
            save_diagnostic_json(report, json_out)
            
            log(f"")
            log(f"  {'='*50}")
            log(f"  RESUMO")
            log(f"  {'='*50}")
            log(f"  Arquivo: {output_path}")
            log(f"  Tamanho: {human_size(report.get('tamanho_disco', 0))}")
            log(f"  Dimensao: {report.get('largura', '?')}x{report.get('altura', '?')}")
            if report.get('gsd_cm'): log(f"  GSD: {report['gsd_cm']:.2f} cm/px")
            problemas = report.get('problemas', [])
            if problemas:
                log(f"  Problemas: {len(problemas)}")
                for p in problemas[:5]: log(f"    - {p}")
            else: log(f"  Problemas: Nenhum")
            log(f"  {'='*50}")
        
        t_total = time.time() - t_inicio
        log(f"")
        log("=" * 70)
        log(f" CONCLUIDO! ({t_total/60:.1f} min)")
        log("=" * 70)
        return

    # ── SE TIVER 2+ ARQUIVOS: MOSAICO COMPLETO ─────────────────────────
    log(f"")
    log("=" * 60)
    log(" MODO MOSAICO: 2+ arquivos - mesclando com feathering")
    log("=" * 60)

    # Converter ECWs para TIFF
    arquivos_processados = []
    temp_dir = tempfile.mkdtemp(prefix="mosaico_ecw_")

    for f in input_files:
        if f.lower().endswith('.ecw'):
            if gdal_cmd:
                tiff_temp = os.path.join(temp_dir, os.path.basename(f).replace('.ecw', '.tif'))
                log(f"")
                log(f"[PRE] Convertendo {os.path.basename(f)}...")
                if convert_ecw_to_tiff(f, tiff_temp, gdal_cmd, gdal_env):
                    arquivos_processados.append(tiff_temp)
                    log(f"[PRE] OK")
                else:
                    log(f"[PRE] ERRO: Pulando {os.path.basename(f)}")
            else:
                log(f"[PRE] ERRO: Sem plugin ECW para ler {os.path.basename(f)}")
        else:
            arquivos_processados.append(f)

    if len(arquivos_processados) == 0:
        log(f"ERRO: Nenhum arquivo processado.")
        shutil.rmtree(temp_dir, ignore_errors=True)
        sys.exit(1)

    log(f"")
    log(f"Prontos para mosaico: {len(arquivos_processados)}")

    # Diagnóstico dos inputs
    if RUN_DIAGNOSTIC_INPUTS:
        log(f"")
        log("=" * 70)
        log(" DIAGNOSTICO ENTRADA")
        log("=" * 70)
        for f in input_files:
            base = os.path.splitext(os.path.basename(f))[0]
            json_path = os.path.join(OUTPUT_DIR, f"diagnostico_input_{base}.json")
            log(f"")
            log(f"Processando: {os.path.basename(f)}")
            save_diagnostic_json(diagnostic_report(f), json_path)

    # Mosaico
    log(f"")
    log("=" * 70)
    log(" MOSAICO")
    log("=" * 70)
    
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    if RUN_MOSAIC:
        if os.path.exists(output_path):
            log(f"ATENCAO: '{output_path}' existe, sobrescrevendo.")
        resultado_path = run_mosaic_tiled(arquivos_processados, output_path)
        if resultado_path:
            output_path = resultado_path
        else:
            log(f"ERRO no mosaico")
            shutil.rmtree(temp_dir, ignore_errors=True)
            sys.exit(1)
    else:
        if not os.path.exists(output_path):
            log(f"AVISO: mosaico desabilitado e saida nao existe")
            shutil.rmtree(temp_dir, ignore_errors=True)
            sys.exit(1)

    # Diagnóstico output
    if RUN_DIAGNOSTIC_OUTPUT and os.path.exists(output_path):
        log(f"")
        log("=" * 70)
        log(" DIAGNOSTICO SAIDA")
        log("=" * 70)
        json_out = os.path.join(OUTPUT_DIR, "diagnostico_output.json")
        log(f"")
        log(f"Processando: {os.path.basename(output_path)}")
        report = diagnostic_report(output_path)
        save_diagnostic_json(report, json_out)

        log(f"")
        log(f"  {'='*50}")
        log(f"  RESUMO")
        log(f"  {'='*50}")
        log(f"  Arquivo: {output_path}")
        log(f"  Tamanho: {human_size(report.get('tamanho_disco', 0))}")
        log(f"  Dimensao: {report.get('largura', '?')}x{report.get('altura', '?')}")
        log(f"  Bandas: {report.get('bandas', '?')}")
        if report.get('gsd_cm'): log(f"  GSD: {report['gsd_cm']:.2f} cm/px")
        if report.get('area_ha'): log(f"  Area: {report['area_ha']:.2f} ha")
        problemas = report.get('problemas', [])
        if problemas:
            log(f"  Problemas: {len(problemas)}")
            for p in problemas[:5]: log(f"    - {p}")
            if len(problemas) > 5: log(f"    ... +{len(problemas)-5}")
        else: log(f"  Problemas: Nenhum")
        log(f"  {'='*50}")

    # Listar output
    log(f"")
    log("=" * 70)
    log(" ARQUIVOS GERADOS")
    log("=" * 70)
    for f in sorted(glob.glob(os.path.join(OUTPUT_DIR, "*"))):
        log(f"  {human_size(os.path.getsize(f)):>8}  {os.path.basename(f)}")

    # Limpar
    if not KEEP_TEMP_TIFFS:
        shutil.rmtree(temp_dir, ignore_errors=True)
        log(f"Temp removido: {temp_dir}")

    t_total = time.time() - t_inicio
    log(f"")
    log("=" * 70)
    log(f" CONCLUIDO! ({t_total/60:.1f} min)")
    log("=" * 70)


if __name__ == "__main__":
    main()