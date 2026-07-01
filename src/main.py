from pathlib import Path
from skimage import color, io 
from skimage.color import gray2rgb
from skimage.filters import gaussian, median, threshold_multiotsu 
from skimage.measure import find_contours, label, regionprops
from skimage.morphology import closing, disk, opening
from skimage.segmentation import clear_border
import matplotlib.pyplot as plt

INPUT_IMAGE = Path(r"..\assets\brain.jpg")
OUTPUT_DIR = Path(r"..\output")

# Lê a imagem e converte para escala de cinza (se necessário)
image = io.imread(INPUT_IMAGE)
if image.ndim == 3:
    gray = color.rgb2gray(image)
else:
    gray = image

# Filtro gaussiano para passa baixas e remoção de ruídos com mediana
gaussian_img = gaussian(gray, sigma=1)
median_img = median(gaussian_img, footprint=disk(3))

# Limiarização por Multi-Otsum, o que garante que vamos isolar apenas as partes mais brilhantes da imagem
thresholds = threshold_multiotsu(median_img, classes=3)
binary_img = median_img > thresholds[1] 

# Operações morfológicas
selem = disk(3)
opened_img = opening(binary_img, selem)
closed_img = closing(opened_img, selem)

# Removemos componentes que tocam a borda da imagem (elimina grande parte do crânio)
cleared_img = clear_border(closed_img)

# Componentes Conexos
labels = label(cleared_img)
regions = regionprops(labels)

# Filtramos as regiões: removemos ruídos muito pequenos e pedaços de crânio remanescentes
# Tumores são massas densas (solidity alta), crânio é fino/curvo (solidity baixa)
valid_regions = [r for r in regions if r.area > 50 and r.solidity > 0.6]

if not valid_regions:
    print("Nenhum tumor detectado.")
else:
    # Pegamos o maior componente das regiões válidas (sem crânio)
    largest_region = max(valid_regions, key=lambda region: region.area)
    print(f"Área do tumor detectado: {largest_region.area}")

    tumor_mask = labels == largest_region.label
    contours = find_contours(tumor_mask, level=0.5)

    original_rgb = gray2rgb(gray)

    fig, ax = plt.subplots(figsize=(7,7))
    ax.imshow(original_rgb)

    for contour in contours:
        ax.plot(
            contour[:,1],
            contour[:,0],
            color="red",
            linewidth=2
        )

    ax.set_title("Tumor Detectado")
    ax.axis("off")

    plt.savefig(
        OUTPUT_DIR / "07_tumor.png",
        dpi=300,
        bbox_inches="tight"
    )