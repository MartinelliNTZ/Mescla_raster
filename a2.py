import rasterio
from rasterio.merge import merge
import glob
import os
import numpy as np
from rasterio.enums import Resampling
from scipy.ndimage import distance_transform_edt

print("=== Início do processo de mosaico com transição suave (feathering manual) ===")

# 1. Listar arquivos de entrada
input_files = glob.glob(r"G:\FS_BIOENERGIA\FAZ_MBOI\mosaicos\MOSAICO\input\*.tif")
input_files = [f for f in input_files if "mosaico_final" not in os.path.basename(f)]

print(f"  Arquivos selecionados: {len(input_files)}")
for f in input_files:
    print(f"    - {f}")

if len(input_files) < 2:
    raise FileNotFoundError("São necessários pelo menos 2 arquivos originais.")

# 2. Mesclar primeiro para obter extensão total e transform do mosaico
datasets = [rasterio.open(f) for f in input_files]

# Verifica se todos têm 4 bandas (RGB + Alpha)
for ds in datasets:
    print(f"  {os.path.basename(ds.name)}: {ds.count} bandas, dtype={ds.dtypes[0]}, CRS={ds.crs}")

# Mescla simples para obter o grid de referência (usa 'first' como base)
mosaic_ref, out_transform = merge(datasets, method='first')
out_crs = datasets[0].crs
out_dtype = datasets[0].dtypes[0]
n_bands, height, width = mosaic_ref.shape
print(f"  Grid do mosaico: {width}x{height}, {n_bands} bandas")

# 3. Reprojetar cada imagem para o grid do mosaico final e aplicar feathering
from rasterio.windows import from_bounds
from rasterio.transform import array_bounds

# Limites do mosaico final
bounds = rasterio.transform.array_bounds(height, width, out_transform)
print(f"  Bounds do mosaico: {bounds}")

# Arrays acumuladores: soma ponderada e soma dos pesos
accum_rgb   = np.zeros((3, height, width), dtype=np.float64)
accum_alpha = np.zeros((height, width),    dtype=np.float64)
accum_w     = np.zeros((height, width),    dtype=np.float64)

for ds in datasets:
    name = os.path.basename(ds.name)
    print(f"  Processando {name}...")

    # Reprojetar para o grid do mosaico
    from rasterio.warp import reproject, Resampling as WarpResampling

    # Arrays de destino para esta imagem
    dst_rgb   = np.zeros((3, height, width), dtype=np.float64)
    dst_alpha = np.zeros((1, height, width), dtype=np.float64)

    # Reprojetar bandas RGB (1,2,3)
    for b in range(1, 4):
        reproject(
            source=rasterio.band(ds, b),
            destination=dst_rgb[b-1],
            src_transform=ds.transform,
            src_crs=ds.crs,
            dst_transform=out_transform,
            dst_crs=out_crs,
            resampling=WarpResampling.bilinear,
        )

    # Reprojetar banda Alpha (banda 4)
    reproject(
        source=rasterio.band(ds, 4),
        destination=dst_alpha[0],
        src_transform=ds.transform,
        src_crs=ds.crs,
        dst_transform=out_transform,
        dst_crs=out_crs,
        resampling=WarpResampling.bilinear,
    )

    alpha_2d = dst_alpha[0]  # shape (height, width)

    # --- FEATHERING: peso = distância euclidiana até a borda do alpha ---
    # Mascara binária: pixel válido onde alpha > 0
    mask_valido = alpha_2d > 0

    if not mask_valido.any():
        print(f"    AVISO: {name} sem pixels válidos após reprojeção, pulando.")
        continue

    # Distância de cada pixel válido até a borda (pixels inválidos ou borda da imagem)
    dist = distance_transform_edt(mask_valido)   # em pixels

    # Normaliza pelo máximo para ter peso 0→1
    dist_max = dist.max()
    if dist_max > 0:
        peso = dist / dist_max
    else:
        peso = mask_valido.astype(np.float64)

    # Acumula: RGB ponderado pelo peso
    for c in range(3):
        accum_rgb[c] += dst_rgb[c] * peso

    accum_alpha += alpha_2d * peso   # alpha também ponderado
    accum_w     += peso

    print(f"    dist_max={dist_max:.1f}px  pixels_validos={mask_valido.sum()}")

# 4. Normalizar pela soma dos pesos
print("  Normalizando...")
valido = accum_w > 0

resultado_rgb   = np.zeros((3, height, width), dtype=out_dtype)
resultado_alpha = np.zeros((height, width),    dtype=out_dtype)

for c in range(3):
    tmp = np.where(valido, accum_rgb[c] / accum_w, 0)
    resultado_rgb[c] = np.clip(tmp, 0, np.iinfo(out_dtype).max).astype(out_dtype)

tmp_alpha = np.where(valido, accum_alpha / accum_w, 0)
resultado_alpha = np.clip(tmp_alpha, 0, np.iinfo(out_dtype).max).astype(out_dtype)

# 5. Empilhar em 4 bandas (RGB + Alpha)
resultado = np.concatenate([
    resultado_rgb,
    resultado_alpha[np.newaxis, ...]
], axis=0)

print(f"  Shape final: {resultado.shape}, dtype={resultado.dtype}")

# 6. Salvar
output_path = r"G:\FS_BIOENERGIA\FAZ_MBOI\mosaicos\MOSAICO\mosaico_final_suave5.tif"
print(f"  Salvando em: {output_path}")

with rasterio.open(
    output_path, 'w',
    driver='GTiff',
    height=height,
    width=width,
    count=4,
    dtype=out_dtype,
    crs=out_crs,
    transform=out_transform,
    compress='lzw',
) as dest:
    dest.write(resultado)

    # Overviews FORA do with de escrita — reabre em modo r+
print("  Construindo overviews...")
with rasterio.open(output_path, 'r+') as dest:
    overviews = [2, 4, 8, 16, 32, 64]
    dest.build_overviews(overviews, Resampling.nearest)
    dest.update_tags(ns='rio_overview', resampling='nearest')

# 7. Fechar datasets originais
for ds in datasets:
    ds.close()

print("=== Processo concluído com sucesso ===")