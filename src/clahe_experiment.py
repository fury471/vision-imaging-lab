import matplotlib.pyplot as plt

from image_histograms import compute_grayscale_histogram
from image_metrics import measure_brightness, measure_contrast
from image_transforms import apply_clahe, histogram_equalization
from image_utils import get_default_image_path, get_project_root, load_color_image, to_grayscale


image_path = get_default_image_path()
output_path = get_project_root() / "outputs" / "clahe_experiment.png"

image_bgr = load_color_image(image_path)
gray_image = to_grayscale(image_bgr)

equalized_image = histogram_equalization(gray_image)
clahe_image = apply_clahe(gray_image, clip_limit=2.0, tile_grid_size=(8, 8))

hist_original, bins = compute_grayscale_histogram(gray_image)
hist_equalized, _ = compute_grayscale_histogram(equalized_image)
hist_clahe, _ = compute_grayscale_histogram(clahe_image)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

axes[0, 0].imshow(gray_image, cmap="gray")
axes[0, 0].set_title("Original")
axes[0, 0].axis("off")

axes[0, 1].imshow(equalized_image, cmap="gray")
axes[0, 1].set_title("Histogram Equalization")
axes[0, 1].axis("off")

axes[0, 2].imshow(clahe_image, cmap="gray")
axes[0, 2].set_title("CLAHE")
axes[0, 2].axis("off")

axes[1, 0].bar(bins[:-1], hist_original, width=1.0, align="edge")
axes[1, 0].set_title("Original Histogram")
axes[1, 0].set_xlim(0, 256)

axes[1, 1].bar(bins[:-1], hist_equalized, width=1.0, align="edge")
axes[1, 1].set_title("Equalized Histogram")
axes[1, 1].set_xlim(0, 256)

axes[1, 2].bar(bins[:-1], hist_clahe, width=1.0, align="edge")
axes[1, 2].set_title("CLAHE Histogram")
axes[1, 2].set_xlim(0, 256)

for ax in axes[1]:
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Pixels")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_path, dpi=150)
plt.show()

print("Original brightness:", measure_brightness(gray_image))
print("Equalized brightness:", measure_brightness(equalized_image))
print("CLAHE brightness:", measure_brightness(clahe_image))

print("Original contrast:", measure_contrast(gray_image))
print("Equalized contrast:", measure_contrast(equalized_image))
print("CLAHE contrast:", measure_contrast(clahe_image))

print("Saved figure to:", output_path)