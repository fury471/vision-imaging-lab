import cv2
import matplotlib.pyplot as plt

from image_metrics import measure_brightness, measure_contrast, measure_sharpness_laplacian
from image_utils import get_default_image_path, get_project_root, load_color_image, to_grayscale


image_path = get_default_image_path()
output_path = get_project_root() / "outputs" / "blur_experiment.png"

image_bgr = load_color_image(image_path)
gray_image = to_grayscale(image_bgr)

blur_light = cv2.GaussianBlur(gray_image, (5, 5), 0)
blur_medium = cv2.GaussianBlur(gray_image, (11, 11), 0)
blur_strong = cv2.GaussianBlur(gray_image, (21, 21), 0)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

axes[0, 0].imshow(gray_image, cmap="gray")
axes[0, 0].set_title("Original")
axes[0, 0].axis("off")

axes[0, 1].imshow(blur_light, cmap="gray")
axes[0, 1].set_title("Light Blur")
axes[0, 1].axis("off")

axes[1, 0].imshow(blur_medium, cmap="gray")
axes[1, 0].set_title("Medium Blur")
axes[1, 0].axis("off")

axes[1, 1].imshow(blur_strong, cmap="gray")
axes[1, 1].set_title("Strong Blur")
axes[1, 1].axis("off")

plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()

print("Original brightness:", measure_brightness(gray_image))
print("Light blur brightness:", measure_brightness(blur_light))
print("Medium blur brightness:", measure_brightness(blur_medium))
print("Strong blur brightness:", measure_brightness(blur_strong))

print("Original contrast:", measure_contrast(gray_image))
print("Light blur contrast:", measure_contrast(blur_light))
print("Medium blur contrast:", measure_contrast(blur_medium))
print("Strong blur contrast:", measure_contrast(blur_strong))

print("Original sharpness:", measure_sharpness_laplacian(gray_image))
print("Light blur sharpness:", measure_sharpness_laplacian(blur_light))
print("Medium blur sharpness:", measure_sharpness_laplacian(blur_medium))
print("Strong blur sharpness:", measure_sharpness_laplacian(blur_strong))

print("Saved figure to:", output_path)