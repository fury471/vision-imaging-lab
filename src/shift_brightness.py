import matplotlib.pyplot as plt
import numpy as np

from image_histograms import compute_grayscale_histogram
from image_metrics import measure_brightness, measure_contrast
from image_utils import get_default_image_path, get_project_root, load_color_image, to_grayscale


image_path = get_default_image_path()
output_path = get_project_root() / "outputs" / "brightness_shift.png"

image_bgr = load_color_image(image_path)
gray_image = to_grayscale(image_bgr)

shift_value = 40

gray_int = gray_image.astype(np.int16)

darker_image = np.clip(gray_int - shift_value, 0, 255).astype(np.uint8)
brighter_image = np.clip(gray_int + shift_value, 0, 255).astype(np.uint8)

hist_original, bins = compute_grayscale_histogram(gray_image)
hist_darker, _ = compute_grayscale_histogram(darker_image)
hist_brighter, _ = compute_grayscale_histogram(brighter_image)

num_clipped_dark = np.sum(gray_int - shift_value < 0)
num_clipped_bright = np.sum(gray_int + shift_value > 255)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

axes[0, 0].imshow(gray_image, cmap="gray")
axes[0, 0].set_title("Original")
axes[0, 0].axis("off")

axes[0, 1].imshow(darker_image, cmap="gray")
axes[0, 1].set_title("Darker")
axes[0, 1].axis("off")

axes[0, 2].imshow(brighter_image, cmap="gray")
axes[0, 2].set_title("Brighter")
axes[0, 2].axis("off")

axes[1, 0].bar(bins[:-1], hist_original, width=1.0, align="edge")
axes[1, 0].set_title("Original Histogram")
axes[1, 0].set_xlim(0, 256)

axes[1, 1].bar(bins[:-1], hist_darker, width=1.0, align="edge")
axes[1, 1].set_title("Darker Histogram")
axes[1, 1].set_xlim(0, 256)

axes[1, 2].bar(bins[:-1], hist_brighter, width=1.0, align="edge")
axes[1, 2].set_title("Brighter Histogram")
axes[1, 2].set_xlim(0, 256)

for ax in axes[1]:
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Pixels")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_path, dpi=150)
plt.show()

print("Original brightness:", measure_brightness(gray_image))
print("Darker brightness:", measure_brightness(darker_image))
print("Brighter brightness:", measure_brightness(brighter_image))

print("Original contrast:", measure_contrast(gray_image))
print("Darker contrast:", measure_contrast(darker_image))
print("Brighter contrast:", measure_contrast(brighter_image))

print("Pixels clipped to 0 in darker image:", num_clipped_dark)
print("Pixels clipped to 255 in brighter image:", num_clipped_bright)

print("Original histogram at 0:", hist_original[0])
print("Darker histogram at 0:", hist_darker[0])
print("Original histogram at 255:", hist_original[255])
print("Brighter histogram at 255:", hist_brighter[255])

print("Saved figure to:", output_path)