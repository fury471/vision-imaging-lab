import matplotlib.pyplot as plt
import pandas as pd

from image_utils import get_project_root


project_root = get_project_root()
csv_path = project_root / "outputs" / "clahe_dataset_evaluation.csv"
output_path = project_root / "outputs" / "clahe_category_summary.png"

df = pd.read_csv(csv_path)

metrics = [
    "brightness_delta",
    "contrast_delta",
    "sharpness_delta",
    "noise_delta",
]

median_summary = (
    df.groupby("category")[metrics]
    .median()
    .round(3)
    .reset_index()
)

categories = median_summary["category"].tolist()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.subplots_adjust(hspace=0.35, wspace=0.25)

for ax, metric in zip(axes.flat, metrics):
    values = median_summary[metric].tolist()
    x_positions = range(len(categories))

    ax.bar(x_positions, values, alpha=0.75)
    ax.set_title(metric.replace("_", " ").title())
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(categories, rotation=15)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.grid(True, axis="y", alpha=0.3)

plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()

print("Saved CLAHE category summary plot to:", output_path)