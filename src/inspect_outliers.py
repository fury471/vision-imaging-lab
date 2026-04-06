import cv2
import matplotlib.pyplot as plt
import pandas as pd

from image_utils import get_project_root, load_color_image


project_root = get_project_root()
csv_path = project_root / "outputs" / "dataset_analysis.csv"
data_root = project_root / "data" / "raw"
output_path = project_root / "outputs" / "outlier_inspection.png"

df = pd.read_csv(csv_path)

top_sharpness = df.sort_values("sharpness", ascending=False).head(4)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.subplots_adjust(hspace=0.35, wspace=0.08)

for ax, (_, row) in zip(axes.flat, top_sharpness.iterrows()):
    image_path = data_root / row["category"] / row["filename"]
    image_bgr = load_color_image(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    ax.imshow(image_rgb)
    ax.set_title(
        f"{row['category']} | {row['filename']}\n"
        f"sharpness={row['sharpness']:.1f}, noise={row['estimated_noise']:.1f}"
    )
    ax.axis("off")

plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()

print("Saved outlier inspection figure to:", output_path)