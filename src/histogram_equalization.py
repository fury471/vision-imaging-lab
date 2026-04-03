import matplotlib.pyplot as plt

from image_histograms import compute_grayscale_histogram
from image_metrics import measure_brightness, measure_contrast
from image_transforms import histogram_equalization
from image_utils import get_default_image_path, get_project_root, load_color_image, to_grayscale


image_path = get_default_image_path()
output_path = get_project_root() / "outputs" / "histogram_equalization.png"

image_bgr = load_color_image(image_path)
gray_image = to_grayscale(image_bgr)

equalized_image = histogram_equalization(gray_image)

hist_original, bins = compute_grayscale_histogram(gray_image)
hist_equalized, _ = compute_grayscale_histogram(equalized_image)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].imshow(gray_image, cmap="gray")
axes[0, 0].set_title("Original")
axes[0, 0].axis("off")

axes[0, 1].imshow(equalized_image, cmap="gray")
axes[0, 1].set_title("Equalized")
axes[0, 1].axis("off")

axes[1, 0].bar(bins[:-1], hist_original, width=1.0, align="edge")
axes[1, 0].set_title("Original Histogram")
axes[1, 0].set_xlim(0, 256)

axes[1, 1].bar(bins[:-1], hist_equalized, width=1.0, align="edge")
axes[1, 1].set_title("Equalized Histogram")
axes[1, 1].set_xlim(0, 256)

for ax in axes[1]:
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Pixels")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_path, dpi=150)
plt.show()

print("Original brightness:", measure_brightness(gray_image))
print("Equalized brightness:", measure_brightness(equalized_image))

print("Original contrast:", measure_contrast(gray_image))
print("Equalized contrast:", measure_contrast(equalized_image))

print("Saved figure to:", output_path)