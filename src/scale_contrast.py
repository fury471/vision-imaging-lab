import cv2
import matplotlib.pyplot as plt

from image_histograms import compute_grayscale_histogram
from image_metrics import (
    measure_brightness,
    measure_contrast,
    measure_min_intensity,
    measure_max_intensity,
)
from image_utils import get_default_image_path, get_project_root, load_color_image, to_grayscale


image_path = get_default_image_path()
output_path = get_project_root() / "outputs" / "contrast_scaling.png"

image_bgr = load_color_image(image_path)
gray_image = to_grayscale(image_bgr)

low_contrast_image = cv2.convertScaleAbs(gray_image, alpha=0.7, beta=0)
high_contrast_image = cv2.convertScaleAbs(gray_image, alpha=1.3, beta=0)

hist_original, bins = compute_grayscale_histogram(gray_image)
hist_low, _ = compute_grayscale_histogram(low_contrast_image)
hist_high, _ = compute_grayscale_histogram(high_contrast_image)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

axes[0, 0].imshow(gray_image, cmap="gray")
axes[0, 0].set_title("Original")
axes[0, 0].axis("off")

axes[0, 1].imshow(low_contrast_image, cmap="gray")
axes[0, 1].set_title("Lower Contrast")
axes[0, 1].axis("off")

axes[0, 2].imshow(high_contrast_image, cmap="gray")
axes[0, 2].set_title("Higher Contrast")
axes[0, 2].axis("off")

axes[1, 0].bar(bins[:-1], hist_original, width=1.0, align="edge")
axes[1, 0].set_title("Original Histogram")
axes[1, 0].set_xlim(0, 256)

axes[1, 1].bar(bins[:-1], hist_low, width=1.0, align="edge")
axes[1, 1].set_title("Lower Contrast Histogram")
axes[1, 1].set_xlim(0, 256)

axes[1, 2].bar(bins[:-1], hist_high, width=1.0, align="edge")
axes[1, 2].set_title("Higher Contrast Histogram")
axes[1, 2].set_xlim(0, 256)

for ax in axes[1]:
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Pixels")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_path, dpi=150)
plt.show()

print("Original brightness:", measure_brightness(gray_image))
print("Low-contrast brightness:", measure_brightness(low_contrast_image))
print("High-contrast brightness:", measure_brightness(high_contrast_image))

print("Original contrast:", measure_contrast(gray_image))
print("Low-contrast contrast:", measure_contrast(low_contrast_image))
print("High-contrast contrast:", measure_contrast(high_contrast_image))

print("Original min/max:", measure_min_intensity(gray_image), measure_max_intensity(gray_image))
print("Low-contrast min/max:", measure_min_intensity(low_contrast_image), measure_max_intensity(low_contrast_image))
print("High-contrast min/max:", measure_min_intensity(high_contrast_image), measure_max_intensity(high_contrast_image))

print("Saved figure to:", output_path)
