import cv2
import matplotlib.pyplot as plt
import numpy as np

from image_utils import get_default_image_path, get_project_root, load_color_image, to_grayscale


image_path = get_default_image_path()
output_path = get_project_root() / "outputs" / "original_vs_gray.png"

image_bgr = load_color_image(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
gray_image = to_grayscale(image_bgr)

print("RGB shape:", image_rgb.shape)
print("Gray shape:", gray_image.shape)
print("Gray min:", np.min(gray_image))
print("Gray max:", np.max(gray_image))
print("Gray mean:", np.mean(gray_image))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(image_rgb)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(gray_image, cmap="gray")
axes[1].set_title("Grayscale Image")
axes[1].axis("off")

plt.tight_layout()
plt.savefig(output_path, dpi=150)
plt.show()

print("Saved comparison figure to:", output_path)
