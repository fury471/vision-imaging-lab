import pandas as pd

from image_metrics import (
    estimate_noise_std,
    measure_brightness,
    measure_contrast,
    measure_sharpness_laplacian,
)
from image_transforms import apply_clahe, scale_contrast_around_mean
from image_utils import get_project_root, load_color_image, to_grayscale


project_root = get_project_root()
data_root = project_root / "data" / "raw"
output_path = project_root / "outputs" / "compare_clahe_vs_centered_contrast.csv"

image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

results = []

for category_dir in sorted(data_root.iterdir()):
    if not category_dir.is_dir():
        continue

    category_name = category_dir.name

    for image_path in sorted(category_dir.iterdir()):
        if image_path.suffix.lower() not in image_extensions:
            continue

        image_bgr = load_color_image(image_path)
        gray_image = to_grayscale(image_bgr)

        centered_image = scale_contrast_around_mean(gray_image, alpha=1.2)
        clahe_image = apply_clahe(gray_image, clip_limit=1.0, tile_grid_size=(12, 12))

        for method_name, transformed in [
            ("centered_contrast", centered_image),
            ("clahe_soft", clahe_image),
        ]:
            result = {
                "category": category_name,
                "filename": image_path.name,
                "method": method_name,
                "brightness_delta": float(measure_brightness(transformed) - measure_brightness(gray_image)),
                "contrast_delta": float(measure_contrast(transformed) - measure_contrast(gray_image)),
                "sharpness_delta": float(measure_sharpness_laplacian(transformed) - measure_sharpness_laplacian(gray_image)),
                "noise_delta": float(estimate_noise_std(transformed) - estimate_noise_std(gray_image)),
            }
            results.append(result)

df = pd.DataFrame(results).round(3)

summary = (
    df.groupby(["category", "method"])[
        ["brightness_delta", "contrast_delta", "sharpness_delta", "noise_delta"]
    ]
    .median()
    .round(3)
    .reset_index()
)

print("\nMedian method comparison by category:\n")
print(summary.to_string(index=False))

df.to_csv(output_path, index=False)
print(f"\nSaved comparison results to: {output_path}")