import matplotlib.pyplot as plt

from image_metrics import measure_brightness, measure_contrast
from image_transforms import apply_clahe
from image_utils import get_default_image_path, get_project_root, load_color_image, to_grayscale


image_path = get_default_image_path()
output_path = get_project_root() / "outputs" / "clahe_tuning.png"

image_bgr = load_color_image(image_path)
gray_image = to_grayscale(image_bgr)

clahe_soft = apply_clahe(gray_image, clip_limit=1.0, tile_grid_size=(12, 12))
clahe_default = apply_clahe(gray_image, clip_limit=2.0, tile_grid_size=(8, 8))
clahe_strong = apply_clahe(gray_image, clip_limit=3.0, tile_grid_size=(6, 6))

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].imshow(gray_image, cmap="gray")
axes[0, 0].set_title("Original")
axes[0, 0].axis("off")

axes[0, 1].imshow(clahe_soft, cmap="gray")
axes[0, 1].set_title("CLAHE Soft")
axes[0, 1].axis("off")

axes[1, 0].imshow(clahe_default, cmap="gray")
axes[1, 0].set_title("CLAHE Default")
axes[1, 0].axis("off")

axes[1, 1].imshow(clahe_strong, cmap="gray")
axes[1, 1].set_title("CLAHE Strong")
axes[1, 1].axis("off")

plt.savefig(output_path, dpi=150)
plt.show()

print("Original brightness:", measure_brightness(gray_image))
print("CLAHE soft brightness:", measure_brightness(clahe_soft))
print("CLAHE default brightness:", measure_brightness(clahe_default))
print("CLAHE strong brightness:", measure_brightness(clahe_strong))

print("Original contrast:", measure_contrast(gray_image))
print("CLAHE soft contrast:", measure_contrast(clahe_soft))
print("CLAHE default contrast:", measure_contrast(clahe_default))
print("CLAHE strong contrast:", measure_contrast(clahe_strong))

print("Saved figure to:", output_path)