import matplotlib.pyplot as plt
import pandas as pd

from image_utils import get_project_root


project_root = get_project_root()
csv_path = project_root / "outputs" / "dataset_analysis.csv"
output_path = project_root / "outputs" / "dataset_summary.png"

df = pd.read_csv(csv_path)

metrics = ["brightness", "contrast", "sharpness", "estimated_noise"]

summary = (
    df.groupby("category")[metrics]
    .mean()
    .round(3)
    .reset_index()
)

categories = summary["category"].tolist()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.subplots_adjust(hspace=0.35, wspace=0.25)

for ax, metric in zip(axes.flat, metrics):
    x_positions = range(len(categories))
    avg_values = summary[metric].tolist()

    ax.bar(x_positions, avg_values, alpha=0.7)

    for i, category in enumerate(categories):
        category_values = df[df["category"] == category][metric].tolist()
        ax.scatter([i] * len(category_values), category_values, color="black", zorder=3)

    ax.set_title(metric.replace("_", " ").title())
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(categories, rotation=15)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.grid(True, axis="y", alpha=0.3)

plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()

print("Saved dataset summary figure to:", output_path)