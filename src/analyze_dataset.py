import pandas as pd

from image_metrics import (
    estimate_noise_std,
    measure_brightness,
    measure_contrast,
    measure_max_intensity,
    measure_min_intensity,
    measure_sharpness_laplacian,
)
from image_utils import get_project_root, load_color_image, to_grayscale


project_root = get_project_root()
data_root = project_root / "data" / "raw"
output_path = project_root / "outputs" / "dataset_analysis.csv"

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

        result = {
            "category": category_name,
            "filename": image_path.name,
            "brightness": float(measure_brightness(gray_image)),
            "contrast": float(measure_contrast(gray_image)),
            "min_intensity": int(measure_min_intensity(gray_image)),
            "max_intensity": int(measure_max_intensity(gray_image)),
            "sharpness": float(measure_sharpness_laplacian(gray_image)),
            "estimated_noise": float(estimate_noise_std(gray_image)),
        }

        results.append(result)

df = pd.DataFrame(results).round(3)

print("\nPer-image results:\n")
print(df.to_string(index=False))

summary = (
    df.groupby("category")[["brightness", "contrast", "sharpness", "estimated_noise"]]
    .mean()
    .round(3)
    .reset_index()
)

print("\nCategory averages:\n")
print(summary.to_string(index=False))

median_summary = (
    df.groupby("category")[["brightness", "contrast", "sharpness", "estimated_noise"]]
    .median()
    .round(3)
    .reset_index()
)

print("\nCategory medians:\n")
print(median_summary.to_string(index=False))


df.to_csv(output_path, index=False)
print(f"\nSaved detailed results to: {output_path}")
