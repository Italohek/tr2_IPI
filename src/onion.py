from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

IMAGE_PATH = Path(r"..\assets\onion.jpg")

# Intervalo de valores de K que serão testados
K_MIN = 2
K_MAX = 10

# Parâmetros para garantir reprodutibilidade e acelerar o cálculo do Silhouette
RANDOM_STATE = 42
SAMPLE_SIZE = 5000

# Lê a imagem
imagem = io.imread(IMAGE_PATH)

# Converte a imagem em uma matriz de pixels, onde cada linha representa um pixel (R, G, B)
pixels = imagem.reshape(-1, 3)

# Listas para armazenar os resultados de cada valor de K
scores = []
segmentacoes = []
valores_k = []

# Testa diferentes quantidades de clusters
for k in range(K_MIN, K_MAX + 1):

    print(f"Testando K = {k}")

    # Cria o modelo K-Means para o valor atual de K
    kmeans = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10
    )

    # Executa o agrupamento dos pixels
    labels = kmeans.fit_predict(pixels)

    # Calcula o Silhouette Score para avaliar a qualidade da segmentação
    score = silhouette_score(
        pixels,
        labels,
        sample_size=SAMPLE_SIZE,
        random_state=RANDOM_STATE
    )

    print(f"Silhouette Score: {score:.4f}")

    # Armazena o valor de K e seu respectivo score
    valores_k.append(k)
    scores.append(score)

    # Reconstrói a imagem substituindo cada pixel pela cor média do cluster ao qual pertence
    imagem_segmentada = kmeans.cluster_centers_[labels]
    imagem_segmentada = imagem_segmentada.reshape(imagem.shape)

    # Converte os valores para uint8 para exibição da imagem
    imagem_segmentada = imagem_segmentada.astype(np.uint8)

    # Salva a imagem segmentada
    segmentacoes.append(imagem_segmentada)

# Identifica o índice correspondente ao maior Silhouette Score
indice_melhor = np.argmax(scores)

# Obtém o melhor valor de K, seu score e a imagem segmentada correspondente
melhor_k = valores_k[indice_melhor]
melhor_score = scores[indice_melhor]
melhor_segmentacao = segmentacoes[indice_melhor]

print(f"\nMelhor K = {melhor_k}")
print(f"Silhouette Score = {melhor_score:.4f}")

# Exibe todas as segmentações obtidas para cada valor de K
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

for ax, img, k, score in zip(axes.ravel(), segmentacoes, valores_k, scores):
    ax.imshow(img)
    ax.set_title(f"K = {k}\nScore = {score:.3f}")
    ax.axis("off")

plt.suptitle("Segmentações utilizando K-Means", fontsize=16)

plt.tight_layout()
plt.show()

# Plota o gráfico do Silhouette Score para cada valor de K testado
plt.figure(figsize=(8, 5))

plt.plot(valores_k, scores, marker='o')

plt.xlabel("Número de clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Escolha do melhor K")

plt.grid(True)

plt.show()

# Exibe lado a lado a imagem original e a melhor segmentação encontrada
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(imagem)
ax[0].set_title("Imagem Original")
ax[0].axis("off")

ax[1].imshow(melhor_segmentacao)
ax[1].set_title(
    f"Melhor Segmentação\nK = {melhor_k} | Score = {melhor_score:.4f}"
)
ax[1].axis("off")

plt.tight_layout()

plt.show()