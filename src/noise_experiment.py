import numpy as np
import matplotlib.pyplot as plt

from image_metrics import estimate_noise_std, measure_brightness, measure_contrast
from image_utils import get_default_image_path, get_project_root, load_color_image, to_grayscale


image_path = get_default_image_path()
output_path = get_project_root() / "outputs" / "noise_experiment.png"

image_bgr = load_color_image(image_path)
gray_image = to_grayscale(image_bgr)

np.random.seed(42)

noise_light = np.random.normal(0, 5, gray_image.shape)
noise_medium = np.random.normal(0, 15, gray_image.shape)
noise_strong = np.random.normal(0, 30, gray_image.shape)

image_light_noise = np.clip(gray_image.astype("float32") + noise_light, 0, 255).astype("uint8")
image_medium_noise = np.clip(gray_image.astype("float32") + noise_medium, 0, 255).astype("uint8")
image_strong_noise = np.clip(gray_image.astype("float32") + noise_strong, 0, 255).astype("uint8")

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.subplots_adjust(hspace=0.25, wspace=0.05)

axes[0, 0].imshow(gray_image, cmap="gray")
axes[0, 0].set_title("Original")
axes[0, 0].axis("off")

axes[0, 1].imshow(image_light_noise, cmap="gray")
axes[0, 1].set_title("Light Noise")
axes[0, 1].axis("off")

axes[1, 0].imshow(image_medium_noise, cmap="gray")
axes[1, 0].set_title("Medium Noise")
axes[1, 0].axis("off")

axes[1, 1].imshow(image_strong_noise, cmap="gray")
axes[1, 1].set_title("Strong Noise")
axes[1, 1].axis("off")

plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()

print("Original brightness:", measure_brightness(gray_image))
print("Light noise brightness:", measure_brightness(image_light_noise))
print("Medium noise brightness:", measure_brightness(image_medium_noise))
print("Strong noise brightness:", measure_brightness(image_strong_noise))

print("Original contrast:", measure_contrast(gray_image))
print("Light noise contrast:", measure_contrast(image_light_noise))
print("Medium noise contrast:", measure_contrast(image_medium_noise))
print("Strong noise contrast:", measure_contrast(image_strong_noise))

print("Original estimated noise:", estimate_noise_std(gray_image))
print("Light estimated noise:", estimate_noise_std(image_light_noise))
print("Medium estimated noise:", estimate_noise_std(image_medium_noise))
print("Strong estimated noise:", estimate_noise_std(image_strong_noise))

print("Saved figure to:", output_path)