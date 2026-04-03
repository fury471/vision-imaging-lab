import matplotlib.pyplot as plt
import numpy as np

from image_histograms import compute_grayscale_histogram
from image_metrics import measure_brightness, measure_contrast
from image_transforms import scale_contrast, scale_contrast_around_mean
from image_utils import get_default_image_path, get_project_root, load_color_image, to_grayscale


image_path = get_default_image_path()
output_path = get_project_root() / "outputs" / "centered_contrast.png"

image_bgr = load_color_image(image_path)
gray_image = to_grayscale(image_bgr)

naive_high_contrast = scale_contrast(gray_image, alpha=1.3)
centered_high_contrast = scale_contrast_around_mean(gray_image, alpha=1.3)

hist_original, bins = compute_grayscale_histogram(gray_image)
hist_naive, _ = compute_grayscale_histogram(naive_high_contrast)
hist_centered, _ = compute_grayscale_histogram(centered_high_contrast)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

axes[0, 0].imshow(gray_image, cmap="gray")
axes[0, 0].set_title("Original")
axes[0, 0].axis("off")

axes[0, 1].imshow(naive_high_contrast, cmap="gray")
axes[0, 1].set_title("Naive High Contrast")
axes[0, 1].axis("off")

axes[0, 2].imshow(centered_high_contrast, cmap="gray")
axes[0, 2].set_title("Centered High Contrast")
axes[0, 2].axis("off")

axes[1, 0].bar(bins[:-1], hist_original, width=1.0, align="edge")
axes[1, 0].set_title("Original Histogram")
axes[1, 0].set_xlim(0, 256)

axes[1, 1].bar(bins[:-1], hist_naive, width=1.0, align="edge")
axes[1, 1].set_title("Naive Contrast Histogram")
axes[1, 1].set_xlim(0, 256)

axes[1, 2].bar(bins[:-1], hist_centered, width=1.0, align="edge")
axes[1, 2].set_title("Centered Contrast Histogram")
axes[1, 2].set_xlim(0, 256)

for ax in axes[1]:
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Pixels")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_path, dpi=150)
plt.show()

print("Original brightness:", measure_brightness(gray_image))
print("Naive brightness:", measure_brightness(naive_high_contrast))
print("Centered brightness:", measure_brightness(centered_high_contrast))

print("Original contrast:", measure_contrast(gray_image))
print("Naive contrast:", measure_contrast(naive_high_contrast))
print("Centered contrast:", measure_contrast(centered_high_contrast))

print("Original mean:", np.mean(gray_image))
print("Saved figure to:", output_path)