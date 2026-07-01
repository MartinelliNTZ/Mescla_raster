# Opções para Escrita de Arquivos ECW

## Status Atual

Seu sistema tem **4 versões do QGIS** instaladas, todas com o plugin ECW da ERDAS, porém **apenas para LEITURA** (consegue abrir arquivos .ecw, mas NÃO consegue criar/salvar novos).

O erro obtido foi:
```
ERROR 1: None of ECW_ENCODE_KEY and ECW_ENCODE_COMPANY were provided.
```

Isso significa que o SDK ECW 5.5 da ERDAS exige **chaves de licença** para criar arquivos ECW. Sem essas chaves, só é possível **ler** ECW.

---

## Opções para Criar/Salvar ECW

### Opção 1 — Usar o GlobalMapper (recomendado)

Você tem o **GlobalMapper 22.1** instalado em:
```
C:\Program Files\GlobalMapper22.1_64bit\
```

O GlobalMapper tem suporte **completo** a ECW (leitura e escrita), pois ele possui sua própria licença do SDK ECW.

**Como usar:**
1. Abra o GlobalMapper
2. File → Open → selecione seu arquivo(s)
3. File → Export → Export Raster/Image Format → escolha "ECW"
4. Ajuste qualidade (TARGET=90) e salve

**Alternativa via script:** Eu posso criar um script que controla o GlobalMapper via comando de linha (o GlobalMapper tem suporte a scripts .gm).

### Opção 2 — Adquirir licença ERDAS ECW SDK

Comprar a licença do SDK ECW da ERDAS (hexagon.com) para habilitar escrita no GDAL/QGIS.

### Opção 3 — Usar formato alternativo (mais prático)

Já que TIFF é o formato que o rasterio manipula nativamente e tem suporte total:
- **Mosaico em TIFF** já funciona perfeitamente (sem precisar de GDAL)
- Compressão DEFLATE + PREDICTOR=2 já reduz o tamanho
- TIFF é aberto, suportado em todo lugar

**Diferença de tamanho:** ECW comprime ~10-20x, TIFF com DEFLATE comprime ~2-5x. Para um TIFF de 33.76 GB:
- ECW (90%): ~2-3 GB
- TIFF DEFLATE: ~8-15 GB

### Opção 4 — Instalar GDAL com ECW de escrita

Desinstalar o QGIS e instalar uma versão que inclua o SDK ECW completo (pago) ou usar OSGeo4W com o pacote gdal-ecw.

---

## O Script `mosaico_merge_ecw_v3.py`

O script atual:
- ✅ **Lê ECW** (usando GDAL do QGIS)
- ✅ **Converte ECW → TIFF** (funcionando, já testado)
- ✅ **Faz mosaico feathering** em blocos (TILE_SIZE=4096)
- ❌ **Escrever ECW** — limitado pela licença

Quando você configura `OUTPUT_FILENAME = "mosaico_final.ecw"` mas o sistema não tem licença de escrita, o script agora:
1. Detecta que ECW não pode ser escrito
2. Altera automaticamente a saída para `.tif`
3. Avisa no terminal

---

## Perguntas

Quer que eu:

A) Crie um script que use o **GlobalMapper em modo linha de comando** para converter TIFF → ECW?

B) Deixe o script atual **apenas com saída TIFF** (sem tentar ECW)?

C) Crie o script usando **outra alternativa** que você conheça?