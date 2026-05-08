import rasterio
from rasterio.warp import reproject, Resampling as WarpResampling
from rasterio.merge import merge
from rasterio.enums import ColorInterp
import glob
import os
import numpy as np
from scipy.ndimage import distance_transform_edt
import warnings
warnings.filterwarnings("ignore")

print("=== Mosaico feathering full-RAM (float32) ===")

# ── 1. Inputs ─────────────────────────────────────────────────────────────────
input_files = glob.glob(r"G:\FS_BIOENERGIA\FAZ_MBOI\mosaicos\MOSAICO\input\*.tif")
input_files = [f for f in input_files if "mosaico_final" not in os.path.basename(f)]
print(f"  Arquivos: {len(input_files)}")
for f in input_files:
    print(f"    - {f}")

# ── 2. Grid de referência ─────────────────────────────────────────────────────
datasets = [rasterio.open(f) for f in input_files]

mosaic_ref, out_transform = merge(datasets, method='first')
out_crs   = datasets[0].crs
out_dtype = datasets[0].dtypes[0]
_, height, width = mosaic_ref.shape
del mosaic_ref
print(f"  Grid: {width}x{height}  dtype={out_dtype}")

iinfo = np.iinfo(out_dtype)

# ── 3. Acumuladores float32 ───────────────────────────────────────────────────
accum_rgb   = np.zeros((3, height, width), dtype=np.float32)
accum_alpha = np.zeros((height, width),    dtype=np.float32)
accum_w     = np.zeros((height, width),    dtype=np.float32)

# ── 4. Para cada imagem ───────────────────────────────────────────────────────
for ds in datasets:
    name = os.path.basename(ds.name)
    print(f"  Processando {name}...")

    # Banda 4 como máscara — ignora o nodata=0 usando dataset_mask
    # dataset_mask: 255=válido, 0=inválido (mais confiável que ler banda 4 diretamente)
    alpha_orig = ds.dataset_mask().astype(np.float32)

    mask = alpha_orig > 0
    dist = distance_transform_edt(mask).astype(np.float32)
    dmax = dist.max()
    if dmax > 0:
        dist /= dmax
    del alpha_orig, mask

    # Reprojeta peso para o grid final
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

    # Lê todas as bandas
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
    print(f"    OK")

# ── 5. Normalizar ─────────────────────────────────────────────────────────────
print("  Normalizando...")
valido = accum_w > 0

resultado = np.zeros((4, height, width), dtype=out_dtype)
tmp = np.zeros((height, width), dtype=np.float32)

for c in range(3):
    np.divide(accum_rgb[c], accum_w, out=tmp, where=valido)
    resultado[c] = np.clip(tmp, iinfo.min, iinfo.max).astype(out_dtype)

np.divide(accum_alpha, accum_w, out=tmp, where=valido)
resultado[3] = np.clip(tmp, iinfo.min, iinfo.max).astype(out_dtype)

del accum_rgb, accum_alpha, accum_w, valido, tmp

# ── 6. Salvar com ColorInterp correto ────────────────────────────────────────
output_path = r"G:\FS_BIOENERGIA\FAZ_MBOI\mosaicos\MOSAICO\mBOI_MOSBAIXO.tif"
print(f"  Salvando: {output_path}")

with rasterio.open(
    output_path, 'w',
    driver='GTiff',
    height=height,
    width=width,
    count=4,
    dtype=out_dtype,
    crs=out_crs,
    transform=out_transform,
    compress='deflate',       # mesmo do original
    predictor=2,              # melhora compressão deflate em imagens uint8
    tiled=True,
    blockxsize=512,
    blockysize=512,
    interleave='pixel',       # mesmo do original
    nodata=None,              # SEM nodata — transparência só pelo alpha
) as dst:
    dst.write(resultado)

    # ── ColorInterp: força R/G/B/Alpha explicitamente ──────────────────────
    dst.colorinterp = (
        ColorInterp.red,
        ColorInterp.green,
        ColorInterp.blue,
        ColorInterp.alpha,      # <-- isso é o que faltava
    )

del resultado

# ── 7. Overviews ──────────────────────────────────────────────────────────────
print("  Overviews...")
with rasterio.open(output_path, 'r+') as dst:
    dst.build_overviews([2, 4, 8, 16, 32, 64], WarpResampling.nearest)
    dst.update_tags(ns='rio_overview', resampling='nearest')

for ds in datasets:
    ds.close()

# ── 8. Relatório final ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("RELATÓRIO DO ARQUIVO DE SAÍDA")
print("="*60)
with rasterio.open(output_path) as ds:
    alpha = ds.read(4)
    validos = (alpha > 0).sum()
    total   = alpha.size
    res_x   = ds.transform.a
    res_y   = abs(ds.transform.e)
    size_mb = os.path.getsize(output_path) / (1024**2)

    print(f"  Arquivo     : {output_path}")
    print(f"  Tamanho px  : {ds.width} x {ds.height}")
    print(f"  Bandas      : {ds.count}  {ds.dtypes}")
    print(f"  ColorInterp : {ds.colorinterp}")
    print(f"  CRS         : {ds.crs}")
    print(f"  Resolução   : {res_x:.4f} x {res_y:.4f} m/px")
    print(f"  Extent      : {ds.bounds}")
    print(f"  Nodata      : {ds.nodata}")
    print(f"  Compressão  : {ds.compression}")
    print(f"  Tiled       : {ds.is_tiled}")
    print(f"  Overviews   : {ds.overviews(1)}")
    print(f"  Alpha válido: {validos:,} / {total:,} px ({100*validos/total:.1f}%)")
    print(f"  Tamanho disk: {size_mb:.1f} MB")
print("="*60)
print("=== Concluído ===")