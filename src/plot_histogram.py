import matplotlib.pyplot as plt
import numpy as np

from image_histograms import compute_grayscale_histogram, count_pixels
from image_utils import get_default_image_path, get_project_root, load_color_image, to_grayscale


image_path = get_default_image_path()
output_path = get_project_root() / "outputs" / "gray_histogram.png"

image_bgr = load_color_image(image_path)
gray_image = to_grayscale(image_bgr)

histogram, bin_edges = compute_grayscale_histogram(gray_image)
pixel_count = count_pixels(gray_image)

print("Histogram shape:", histogram.shape)
print("Total pixel count from histogram:", np.sum(histogram))
print("Image pixel count:", pixel_count)

plt.figure(figsize=(12, 5))
plt.bar(bin_edges[:-1], histogram, width=1.0, align="edge")
plt.title("Grayscale Intensity Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Number of Pixels")
plt.xlim(0, 256)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_path, dpi=150)
plt.show()


print("Saved histogram to:", output_path)
