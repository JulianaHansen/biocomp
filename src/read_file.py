import logging
import numpy as np
import tifffile as tiff
import pandas as pd
import os

logger = logging.getLogger(__name__)
output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(output_dir, exist_ok=True)


def generate_matrix(file_path, output_filename, block_rows=512):
    '''
    Gera matriz X|Y|R|G|B em CSV
    '''
    
    # abrir arquivo TIFF
    try:
        with tiff.TiffFile(file_path) as tif:
            img = tif.asarray()
        logger.info(f"Processando imagem: {img.shape}")
    except Exception as e:
        logger.exception("Erro ao ler TIFF: %s", e)
        raise

    '''
    Imagens TIFF podem vir em formatos diferentes:
    Grayscale (2D): shape (altura, largura) — só um canal
    RGB (3D): shape (altura, largura, 3) — 3 canais (R, G, B)
    RGBA (3D): shape (altura, largura, 4) — 4 canais (R, G, B, Alpha)
    '''
    if img.ndim == 3 and img.shape[2] == 4: # RGB (3D) com Alpha, descartar Alpha
        img = img[..., :3]
    elif img.ndim == 2: # Grayscale (2D)
        raise ValueError("Imagem em Grayscale (2D).")
    elif img.ndim != 3 or img.shape[2] < 3:
        raise ValueError(f"Formato inesperado: shape={img.shape}")

    altura, largura, _ = img.shape
    logger.info("Dimensões da imagem: %d x %d pixels", altura, largura)

    output_path = os.path.join(output_dir, output_filename)
    logger.info("Salvando blocos em %s", output_path)
    

    modo = 'w'
    # Usa processamento em blocos para evitar uso excessivo de memória.
    for i, y_start in enumerate(range(0, altura, block_rows)):
        y_end = min(altura, y_start + block_rows)
        block_h = y_end - y_start
        
        logger.debug("Processando bloco %d y=[%d:%d]", i, y_start, y_end)
        
        # ler apenas o bloco
        block_rgb = np.asarray(img[y_start:y_end, :, :3], dtype=np.uint8)
        
        # gerar coordenadas apenas para este bloco
        xs = np.tile(np.arange(largura), block_h)
        ys = np.repeat(np.arange(y_start, y_end), largura)
        rgb_flat = block_rgb.reshape(-1, 3)
        
        # montar matriz [X | Y | R | G | B]
        matriz_bloco = np.column_stack((xs, ys, rgb_flat))
        df_bloco = pd.DataFrame(matriz_bloco, columns=['x', 'y', 'r', 'g', 'b'])
        
        if modo == 'w':
            df_bloco.to_csv(output_path, mode=modo, header=True, index=False)
            modo = 'a'
        else:
            df_bloco.to_csv(output_path, mode=modo, header=False, index=False)
        
        logger.info("Bloco %d salvo", i)
    
    logger.info("Matriz completa salva em: %s", output_path)
    print(f"Matriz completa salva em {output_path}")
    return output_path