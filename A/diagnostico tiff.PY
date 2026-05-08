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

REQUISITOS:
pip install rasterio numpy

USO:
python validar_raster.py "D:/meu_raster.tif"
"""

import os
import sys
import math
import json
import rasterio
from rasterio.enums import Compression


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


def main(path):

    if not os.path.exists(path):
        print(f"Arquivo nao encontrado: {path}")
        return

    file_size = os.path.getsize(path)

    print("=" * 80)
    print("VALIDADOR DE RASTER")
    print("=" * 80)

    print(f"\nArquivo: {path}")
    print(f"Tamanho em disco: {human_size(file_size)}")

    with rasterio.open(path) as ds:

        width = ds.width
        height = ds.height
        count = ds.count
        dtypes = ds.dtypes
        crs = ds.crs
        transform = ds.transform
        nodata = ds.nodata
        bounds = ds.bounds
        res_x = abs(transform.a)
        res_y = abs(transform.e)

        driver = ds.driver
        compression = compression_name(ds.compression)

        tiled = ds.is_tiled
        block_shapes = ds.block_shapes
        profile = ds.profile
        predictor = profile.get("predictor") or ds.tags(ns="IMAGE_STRUCTURE").get(
            "PREDICTOR"
        )
        photometric = ds.tags(ns="TIFF").get("PHOTOMETRIC") or ds.tags().get(
            "PHOTOMETRIC"
        )
        overview_levels = ds.overviews(1)
        num_overviews = len(overview_levels)
        has_alpha = count == 4

        # bits por pixel
        dtype = dtypes[0]
        bytes_per_pixel = {
            "uint8": 1,
            "int8": 1,
            "uint16": 2,
            "int16": 2,
            "uint32": 4,
            "int32": 4,
            "float32": 4,
            "float64": 8,
        }.get(dtype, 4)

        tile_w, tile_h = block_shapes[0] if block_shapes else (None, None)
        tile_bytes = (
            tile_w * tile_h * count * bytes_per_pixel if tile_w and tile_h else 0
        )
        tile_mem = human_size(tile_bytes)

        # tamanho teorico sem compressao
        raw_size = width * height * count * bytes_per_pixel

        compression_ratio = raw_size / file_size if file_size > 0 else 0

        megapixels = (width * height) / 1_000_000

        print("\n" + "=" * 80)
        print("METADADOS")
        print("=" * 80)

        print(f"Driver: {driver}")
        print(f"Largura: {width:,}")
        print(f"Altura: {height:,}")
        print(f"Bandas: {count}")
        print(f"Megapixels: {megapixels:.2f} MP")
        print(f"Tipo de pixel: {dtype}")
        print(f"Resolucao X: {res_x}")
        print(f"Resolucao Y: {res_y}")
        print(f"CRS: {crs}")
        print(f"NoData: {nodata}")

        print("\nBounds:")
        print(bounds)

        print("\n" + "=" * 80)
        print("ARMAZENAMENTO")
        print("=" * 80)

        print(f"Compressao: {compression}")
        print(f"Predictor: {predictor or 'NAO informado'}")
        print(f"Tiled: {yes_no(tiled)}")
        print(f"Block shapes: {block_shapes}")
        if tile_w and tile_h:
            print(f"Tamanho de leitura por bloco: {tile_w}x{tile_h} -> {tile_mem}")
        print(f"Photometric: {photometric}")
        print(
            f"Overviews: {num_overviews} nivel(is) {overview_levels if overview_levels else 'nenhum'}"
        )

        print(f"\nTamanho teorico sem compressao: {human_size(raw_size)}")
        print(f"Tamanho real em disco: {human_size(file_size)}")
        print(f"Taxa compressao: {compression_ratio:.2f}x")

        print("\n" + "=" * 80)
        print("CHECKLIST DE VALIDACAO")
        print("=" * 80)

        problems = []

        # FLOAT32
        if dtype == "float32":
            problems.append(
                "Raster FLOAT32 detectado -> MUITO pesado. "
                "Ortomosaicos RGB normalmente deveriam ser UINT8."
            )

        if dtype == "float64":
            problems.append("Raster FLOAT64 detectado -> extremamente pesado.")

        # SEM compressao
        if compression in ["NONE", "None"]:
            problems.append("Raster sem compressao.")

        # nao tiled
        if not tiled:
            problems.append("Raster nao esta tiled.")

        # RGB muito grande
        if count >= 3 and dtype == "uint16":
            problems.append(
                "RGB em UINT16 -> normalmente DJI Terra exporta muito pesado."
            )

        # tamanho absurdo
        if file_size > 20 * 1024**3:
            problems.append("Arquivo acima de 20 GB.")

        # overview
        if num_overviews == 0:
            problems.append(
                "Sem overviews internos. QGIS precisará processar resolução total para zoom, o que causa lentidão."
            )

        # predictor
        if predictor is None and any(
            x in compression.lower() for x in ["deflate", "lzw"]
        ):
            problems.append(
                "Compressão sem PREDICTOR configurado; usar PREDICTOR=2 melhora desempenho para imagens multibanda."
            )

        # alpha / 4 bandas
        if has_alpha:
            problems.append(
                "Raster possui 4 bandas (provavelmente inclui alpha), o que pode reduzir a performance de renderização no QGIS."
            )

        # resolução extrema
        if res_x < 0.01:
            problems.append("Resolucao espacial extremamente alta.")

        if len(problems) == 0:
            print("Nenhum problema grave detectado.")
        else:
            for i, p in enumerate(problems, 1):
                print(f"{i}. {p}")

        print("\n" + "=" * 80)
        print("RECOMENDACOES")
        print("=" * 80)

        print("\n" + "=" * 80)
        print("COMANDO GDAL RECOMENDADO")
        print("=" * 80)

        print(rf"""
gdal_translate "{path}" "{path[:-4]}_OTIMIZADO.tif" ^
    -co COMPRESS=LZW ^
    -co PREDICTOR=2 ^
    -co TILED=YES ^
    -co BIGTIFF=YES
""")

        if dtype == "uint16" and count >= 3:
            print("\nSUGESTAO EXTRA:")
            print("""
Converter RGB uint16 -> uint8 pode reduzir MUITO o tamanho.
Exemplo:
gdal_translate -ot Byte ...
""")

        print("\nFINALIZADO.")


if __name__ == "__main__":

    if len(sys.argv) > 1:
        main(sys.argv[1])
        sys.exit(0)

    # procura automaticamente TIFF na pasta ./A0
    source_dir = os.path.join(".", "A")
    tif_files = [
        os.path.join(source_dir, f)
        for f in os.listdir(source_dir)
        if f.lower().endswith((".tif", ".tiff"))
    ]

    if len(tif_files) == 0:
        print("Nenhum TIFF encontrado na pasta ./A.")
    elif len(tif_files) == 1:
        print(f"Raster encontrado automaticamente: {tif_files[0]}")
        main(tif_files[0])
    else:
        print("Multiplos TIFFs encontrados:\n")

        for i, f in enumerate(tif_files):
            print(f"[{i}] {os.path.basename(f)}")

        idx = int(input("\nEscolha o indice: "))
        main(tif_files[idx])
