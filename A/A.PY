import os
import glob
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge

print("=== Início do processo de mosaico ===")
print("1. Listar os arquivos .tif")
input_dir = r"G:\FS_BIOENERGIA\FAZ_MBOI\mosaicos\MOSAICO"
output_name = "mosaico_final_suave3over.tif"
output_path = os.path.join(input_dir, output_name)
ignore_names = {
    output_name,
    "mosaico_final_suave.tif",
    "mosaico_final_suave2.tif",
    "mosaico_final_suave_overwiew.tif"
}

input_files = glob.glob(os.path.join(input_dir, "*.tif"))
print(f"  arquivos encontrados: {len(input_files)}")
for file in input_files:
    marker = ""
    if os.path.basename(file) in ignore_names:
        marker = " [ignorado]"
    print(f"    - {file}{marker}")

valid_files = [file for file in input_files if os.path.basename(file) not in ignore_names]
if not valid_files:
    raise FileNotFoundError("Nenhum arquivo .tif válido para mesclar foi encontrado na pasta especificada")

print("2. Abrir os datasets")
datasets_to_merge = []
for file in valid_files:
    print(f"  abrindo: {file}")
    try:
        ds = rasterio.open(file)
    except Exception as exc:
        print(f"    erro ao abrir {file}: {exc}")
        continue
    print(f"    bands={ds.count}, crs={ds.crs}, nodata={ds.nodata}, dtype={ds.dtypes[0]}")
    datasets_to_merge.append(ds)

if not datasets_to_merge:
    raise RuntimeError("Nenhum dataset válido pôde ser aberto para mesclagem")

print("3. Mesclar os datasets")
print("  método: first (overlap policy padrão)")
mosaic, out_transform = merge(
    datasets_to_merge,
    method='first'
)
print(f"  mosaico criado: shape={mosaic.shape}")
print(f"  transform={out_transform}")

print("4. Salvar o resultado")
print(f"  caminho de saída: {output_path}")
with rasterio.open(
    output_path, 'w',
    driver='GTiff',
    height=mosaic.shape[1],
    width=mosaic.shape[2],
    count=mosaic.shape[0],
    dtype=mosaic.dtype,
    crs=datasets_to_merge[0].crs,
    transform=out_transform
) as dest:
    dest.write(mosaic)
    overviews = [2, 4, 8, 16, 32, 64]
    print(f"  construindo overviews: {overviews}")
    dest.build_overviews(overviews, Resampling.nearest)
    dest.update_tags(ns='rio_overview', resampling='nearest')
print("  arquivo salvo com sucesso")

print("5. Fechar os datasets")
for ds in datasets_to_merge:
    print(f"  fechando: {ds.name}")
    ds.close()

print("=== Processo concluído ===")
