import rasterio
import glob, os

input_files = glob.glob(r"G:\FS_BIOENERGIA\FAZ_MBOI\mosaicos\MOSAICO\input\*.tif")
input_files = [f for f in input_files if "mosaico_final" not in os.path.basename(f)]

for f in input_files:
    with rasterio.open(f) as ds:
        print(f"\n{'='*60}")
        print(f"Arquivo : {os.path.basename(f)}")
        print(f"Driver  : {ds.driver}")
        print(f"Tamanho : {ds.width}x{ds.height} px")
        print(f"Bandas  : {ds.count}  dtypes={ds.dtypes}")
        print(f"CRS     : {ds.crs}")
        print(f"Transform: {ds.transform}")
        print(f"Nodata  : {ds.nodata}")
        print(f"Compres.: {ds.compression}")
        print(f"Interl. : {ds.interleaving}")
        print(f"Tags    : {ds.tags()}")
        #print(f"ColorInterp: {[ds.colorinterp(b) for b in range(1, ds.count+1)]}")
        print(f"Overviews banda1: {ds.overviews(1)}")
        # Amostra de pixels não-zero no alpha
        alpha = ds.read(ds.count)
        total = alpha.size
        validos = (alpha > 0).sum()
        print(f"Alpha: {validos}/{total} px válidos ({100*validos/total:.1f}%)")