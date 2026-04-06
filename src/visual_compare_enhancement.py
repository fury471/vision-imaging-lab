import matplotlib.pyplot as plt

from image_transforms import apply_clahe, scale_contrast_around_mean
from image_utils import get_project_root, load_color_image, to_grayscale


project_root = get_project_root()
data_root = project_root / "data" / "raw"
output_path = project_root / "outputs" / "visual_compare_enhancements.png"

selected_images = [
    ("low_light", "low_light_01.jpg"),
    ("high_contrast", "high_contrast_01.jpg"),
    ("blurry", "blurry_01.jpg"),
]

fig, axes = plt.subplots(len(selected_images), 3, figsize=(12, 12))
fig.subplots_adjust(hspace=0.35, wspace=0.08)

for row_idx, (category, filename) in enumerate(selected_images):
    image_path = data_root / category / filename
    image_bgr = load_color_image(image_path)
    gray_image = to_grayscale(image_bgr)

    centered_image = scale_contrast_around_mean(gray_image, alpha=1.2)
    clahe_image = apply_clahe(gray_image, clip_limit=1.0, tile_grid_size=(12, 12))

    axes[row_idx, 0].imshow(gray_image, cmap="gray")
    axes[row_idx, 0].set_title(f"{category} | Original")
    axes[row_idx, 0].axis("off")

    axes[row_idx, 1].imshow(centered_image, cmap="gray")
    axes[row_idx, 1].set_title("Centered Contrast")
    axes[row_idx, 1].axis("off")

    axes[row_idx, 2].imshow(clahe_image, cmap="gray")
    axes[row_idx, 2].set_title("CLAHE Soft")
    axes[row_idx, 2].axis("off")

plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()

print("Saved enhancement comparison figure to:", output_path)