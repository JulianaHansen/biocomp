'''
Aqui iremos segmentar a imagem em regiões, que serão as células.
Existem diversars formas de fazer essa segmentação com Clustering e Deep Learning.
'''

import logging
import os
from cellpose import models, io, utils
import cv2
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage import io
from skimage.transform import rescale
from skimage.color import rgb2gray
from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
from skimage import data
from skimage.io import imsave

logger = logging.getLogger(__name__)
output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(output_dir, exist_ok=True)


'''
Clustering
'''

def preprocess_image(image_path, scale_factor=0.2, sigma=2):
    """
    Carrega a imagem, converte para cinza, aplica blur e redimensiona.
    Essencial para Spectral Clustering não travar a memória.
    """
    print(f"--> Carregando imagem: {image_path}")
    # Lê a imagem
    orig_img = io.imread(image_path)
    
    # Se for colorida (3 canais), converte para cinza (necessário para este método de grafo)
    if orig_img.ndim == 3:
        orig_img = rgb2gray(orig_img)
    
    # Aplica filtro Gaussiano para suavizar ruídos antes de reduzir
    smoothened_img = gaussian_filter(orig_img, sigma=sigma)
    
    # Reduz a imagem (Downsampling)
    # mode='reflect' e anti_aliasing=False são do exemplo original para preservar bordas
    rescaled_img = rescale(smoothened_img, scale_factor, mode="reflect", anti_aliasing=False)
    
    print(f"--> Imagem redimensionada de {orig_img.shape} para {rescaled_img.shape}")
    temp_image_path = os.path.join(output_dir, "rescaled_img.png")
    imsave(temp_image_path, rescaled_img)
    return temp_image_path

def compare_spectral_methods(image_path, n_clusters=20, beta=10):
    """
    Constrói o grafo da imagem e compara 3 métodos de atribuição de labels:
    kmeans, discretize e cluster_qr.
    """
    img = preprocess_image(image_path)
    print("--> Construindo o grafo da imagem.")
    # Converte a imagem em um grafo com valores de gradiente nas arestas
    graph = image.img_to_graph(img)

    # Função decrescente do gradiente: exponencial
    # Beta baixo = segmentação independente da imagem
    # Beta alto = segmentação segue as bordas da imagem
    eps = 1e-6
    graph.data = np.exp(-beta * graph.data / graph.data.std()) + eps
    
    # Lista de métodos para comparar
    methods = ["kmeans", "discretize", "cluster_qr"]
    
    # Prepara a plotagem
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(f'Comparação Spectral Clustering (n_clusters={n_clusters})', fontsize=16)

    results = {}

    for i, assign_labels in enumerate(methods):
        print(f"\nRodando método: {assign_labels}...")
        
        t0 = time.time()
        
        # O passo pesado: Spectral Clustering
        labels = spectral_clustering(
            graph,
            n_clusters=n_clusters,
            eigen_tol=1e-7,
            assign_labels=assign_labels,
            random_state=42,
            eigen_solver='arpack' # Solver padrão robusto
        )
        
        t1 = time.time()
        dt = t1 - t0
        results[assign_labels] = dt
        
        # Redimensiona os labels para o formato da imagem
        labels = labels.reshape(img.shape)

        # Visualização
        ax = axes[i]
        ax.imshow(img, cmap=plt.cm.gray)
        ax.set_xticks(())
        ax.set_yticks(())
        
        title = f"{assign_labels}\nTempo: {dt:.2f}s"
        ax.set_title(title)
        print(f"Concluído: {title}")

        # Desenha os contornos de cada cluster
        # Loop para colorir cada região encontrada
        for l in range(n_clusters):
            # Gera uma cor espectral baseada no índice
            color = [plt.cm.nipy_spectral((l + 1) / float(n_clusters + 1))]
            ax.contour(labels == l, colors=color, linewidths=1)

    plt.tight_layout()
    plt.show()
    print("\n=== Resumo dos Tempos ===")
    for metodo, tempo in results.items():
        print(f"{metodo}: {tempo:.4f} segundos")
    return results

'''
Deep Learning:
Existem grandes modelos pré-treinados específicos para bioimagem.
'''

def cellpose_segmentation(image_path, diameter=None):
    '''
    O Cellpose é uma rede neural pré-treinada com milhares de imagens de células.
    Em vez de apenas classificar pixels, prevê campos de vetores (setas) que apontam para o centro da célula.
    Mesmo se duas células estiverem coladas, os vetores de uma apontam para a esquerda e os da outra para a direita.
    O algoritmo segue as setas e sabe separar as duas perfeitamente
    '''
    img = io.imread(image_path)

    # Verifica dinamicamente se existe GPU disponível
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        logger.info("Usando GPU para segmentação com Cellpose.")
    else:
        logger.info("Usando CPU para segmentação com Cellpose.")
    
    # Carrega o modelo 'cyto' (padrão para citoplasma + núcleo)
    # gpu=True tenta usar CUDA automaticamente se disponível
    model = models.CellposeModel(gpu=use_gpu, model_type='cyto')
    
    # Executa a segmentação
    # net_avg=False: O Cellpose roda a rede neural na imagem original, depois rotacionada, depois invertida, etc., e faz a média dos resultados para garantir que pegou tudo.
    # Com net_avg=False ele roda uma única vez.
    masks, _, _, _ = model.eval(img, 
                                diameter=diameter, 
                                tile=True, # Divide a imagem em pedaços e remonta o resultado.
                                channels=[0, 0], # Modo automático para imagens cinza ou RGB.
                                net_avg=use_gpu) # OTIMIZAÇÃO - Desativa a média de redes.

    # Gera os contornos (retorna True onde é borda)
    outlines = utils.masks_to_outlines(masks)
    
    # Copia a imagem original para desenhar em cima
    output_img = img.copy()
    
    # Se a imagem for Preto e Branco (2D), converte para RGB para aceitar cor no contorno
    if output_img.ndim == 2:
        output_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2RGB)
        
    # Pinta os contornos de AMARELO [255, 255, 0]
    # (Pode mudar para [0, 255, 0] se preferir Verde)
    output_img[outlines] = [255, 255, 0]

    # OpenCV usa BGR, Cellpose usa RGB. Convertemos antes de salvar.
    cv2.imwrite(output_dir, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
    print(f"Salvo em: {output_dir}")
    
    return masks

def stardist_segmentation():
    '''
    O StarDist assume que células são objetos redondos/ovais.
    Ele é matematicamente forçado a criar formas convexas.
    Ele é extremamente preciso para contagem em tecidos densos.
    '''
    pass

def cellpose_gui():
    '''
    Se o Cellpose e o StarDist falharem, é possível treinar a própria rede manualmente com o Cellpose GUI (interface gráfica):
    1. O Cellpose faz uma predição inicial (que pode ter erros).
    2. Você corrige manualmente no GUI (apaga um falso positivo, desenha uma célula que faltou).
    3. Você clica em "Train".
    4. O modelo aprende com seu erro instantaneamente.
    '''
    pass