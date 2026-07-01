from pathlib import Path

import matplotlib.pyplot as plt
from skimage import io, color


# ----------------------------
# Diretórios
# ----------------------------
INPUT_IMAGE = "brain.jpg"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ----------------------------
# Leitura da imagem
# ----------------------------
image = io.imread(INPUT_IMAGE)

print(f"Formato da imagem: {image.shape}")

# ----------------------------
# Conversão para escala de cinza
# ----------------------------
if image.ndim == 3:
    gray = color.rgb2gray(image)
    print("Imagem convertida para escala de cinza.")
else:
    gray = image
    print("Imagem já está em escala de cinza.")

# ----------------------------
# Salvar resultado
# ----------------------------
io.imsave(
    OUTPUT_DIR / "01_gray.png",
    (gray * 255).astype("uint8")
)

# ----------------------------
# Exibir
# ----------------------------
plt.figure(figsize=(6, 6))
plt.imshow(gray, cmap="gray")
plt.title("Imagem em escala de cinza")
plt.axis("off")
plt.show()