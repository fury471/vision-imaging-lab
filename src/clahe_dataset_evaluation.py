import pandas as pd

from image_metrics import (
    estimate_noise_std,
    measure_brightness,
    measure_contrast,
    measure_sharpness_laplacian,
)
from image_transforms import apply_clahe
from image_utils import get_project_root, load_color_image, to_grayscale


project_root = get_project_root()
data_root = project_root / "data" / "raw"
output_path = project_root / "outputs" / "clahe_dataset_evaluation.csv"

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

        clahe_image = apply_clahe(gray_image, clip_limit=1.0, tile_grid_size=(12, 12))

        brightness_before = float(measure_brightness(gray_image))
        brightness_after = float(measure_brightness(clahe_image))

        contrast_before = float(measure_contrast(gray_image))
        contrast_after = float(measure_contrast(clahe_image))

        sharpness_before = float(measure_sharpness_laplacian(gray_image))
        sharpness_after = float(measure_sharpness_laplacian(clahe_image))

        noise_before = float(estimate_noise_std(gray_image))
        noise_after = float(estimate_noise_std(clahe_image))

        result = {
            "category": category_name,
            "filename": image_path.name,
            "brightness_before": brightness_before,
            "brightness_after": brightness_after,
            "brightness_delta": brightness_after - brightness_before,
            "contrast_before": contrast_before,
            "contrast_after": contrast_after,
            "contrast_delta": contrast_after - contrast_before,
            "sharpness_before": sharpness_before,
            "sharpness_after": sharpness_after,
            "sharpness_delta": sharpness_after - sharpness_before,
            "noise_before": noise_before,
            "noise_after": noise_after,
            "noise_delta": noise_after - noise_before,
        }

        results.append(result)

df = pd.DataFrame(results).round(3)

print("\nPer-image CLAHE effects:\n")
print(
    df[
        [
            "category",
            "filename",
            "brightness_delta",
            "contrast_delta",
            "sharpness_delta",
            "noise_delta",
        ]
    ].to_string(index=False)
)

summary = (
    df.groupby("category")[
        [
            "brightness_delta",
            "contrast_delta",
            "sharpness_delta",
            "noise_delta",
        ]
    ]
    .mean()
    .round(3)
    .reset_index()
)

print("\nCategory average CLAHE effects:\n")
print(summary.to_string(index=False))

median_summary = (
    df.groupby("category")[
        [
            "brightness_delta",
            "contrast_delta",
            "sharpness_delta",
            "noise_delta",
        ]
    ]
    .median()
    .round(3)
    .reset_index()
)

print("\nCategory median CLAHE effects:\n")
print(median_summary.to_string(index=False))

df.to_csv(output_path, index=False)
print(f"\nSaved detailed CLAHE evaluation to: {output_path}")
