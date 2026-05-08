import rasterio
from rasterio.merge import merge
import glob

print("=== Início do processo de mosaico ===")
print("1. Listar os arquivos .tif")
input_files = glob.glob(r"G:\FS_BIOENERGIA\FAZ_MBOI\mosaicos\MOSAICO\*.tif")
print(f"  arquivos encontrados: {len(input_files)}")
for file in input_files:
    print(f"    - {file}")

if not input_files:
    raise FileNotFoundError("Nenhum arquivo .tif encontrado na pasta especificada")

print("2. Abrir os datasets")
datasets_to_merge = []
for file in input_files:
    print(f"  abrindo: {file}")
    ds = rasterio.open(file)
    print(f"    bands={ds.count}, crs={ds.crs}, nodata={ds.nodata}, dtype={ds.dtypes[0]}")
    datasets_to_merge.append(ds)

print("3. Mesclar os datasets")
print("  método: first (overlap policy padrão)")
mosaic, out_transform = merge(
    datasets_to_merge,
    method='first'
)
print(f"  mosaico criado: shape={mosaic.shape}")
print(f"  transform={out_transform}")

print("4. Salvar o resultado")
output_path = "mosaico_final_suave2.tif"
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
print("  arquivo salvo com sucesso")

print("5. Fechar os datasets")
for ds in datasets_to_merge:
    print(f"  fechando: {ds.name}")
    ds.close()

print("=== Processo concluído ===")
