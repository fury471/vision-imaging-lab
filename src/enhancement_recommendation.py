import pandas as pd

from image_utils import get_project_root


project_root = get_project_root()
csv_path = project_root / "outputs" / "clahe_dataset_evaluation.csv"

df = pd.read_csv(csv_path)

summary = (
    df.groupby(["category"])[
        ["brightness_delta", "contrast_delta", "sharpness_delta", "noise_delta"]
    ]
    .median()
    .round(3)
    .reset_index()
)

print("Current heuristic recommendations:\n")

for _, row in summary.iterrows():
    category = row["category"]

    if category == "low_light":
        recommendation = "Soft CLAHE is the preferred first try because it improves visibility most."
    elif category == "high_contrast":
        recommendation = "Centered contrast is preferred because CLAHE adds little useful contrast."
    elif category == "blurry":
        recommendation = "Neither method fixes blur; enhancement may help visibility slightly, but blur remains the main issue."
    else:
        recommendation = "Use visual inspection and metrics together before choosing a transform."

    print(f"{category}: {recommendation}")