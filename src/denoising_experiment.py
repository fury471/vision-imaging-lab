import numpy as np
import matplotlib.pyplot as plt

from image_metrics import (
    estimate_noise_std,
    measure_brightness,
    measure_contrast,
    measure_sharpness_laplacian,
    measure_mae,
    measure_mse,
    measure_psnr,
)

from image_transforms import gaussian_denoise, median_denoise, bilateral_denoise
from image_utils import get_default_image_path, get_project_root, load_color_image, to_grayscale


image_path = get_default_image_path()
output_path = get_project_root() / "outputs" / "denoising_experiment.png"

image_bgr = load_color_image(image_path)
gray_image = to_grayscale(image_bgr)

np.random.seed(42)
noise = np.random.normal(0, 15, gray_image.shape)
noisy_image = np.clip(gray_image.astype("float32") + noise, 0, 255).astype("uint8")

gaussian_image = gaussian_denoise(noisy_image, kernel_size=(5, 5), sigma=0)
median_image = median_denoise(noisy_image, kernel_size=5)
bilateral_image = bilateral_denoise(noisy_image, d=9, sigma_color=50, sigma_space=50)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.subplots_adjust(hspace=0.25, wspace=0.05)

axes[0, 0].imshow(noisy_image, cmap="gray")
axes[0, 0].set_title("Noisy")
axes[0, 0].axis("off")

axes[0, 1].imshow(gaussian_image, cmap="gray")
axes[0, 1].set_title("Gaussian Denoise")
axes[0, 1].axis("off")

axes[1, 0].imshow(median_image, cmap="gray")
axes[1, 0].set_title("Median Denoise")
axes[1, 0].axis("off")

axes[1, 1].imshow(bilateral_image, cmap="gray")
axes[1, 1].set_title("Bilateral Denoise")
axes[1, 1].axis("off")

plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()

for name, image in [
    ("Noisy", noisy_image),
    ("Gaussian", gaussian_image),
    ("Median", median_image),
    ("Bilateral", bilateral_image),
]:
    print(f"{name} brightness:", measure_brightness(image))
    print(f"{name} contrast:", measure_contrast(image))
    print(f"{name} estimated noise:", estimate_noise_std(image))
    print(f"{name} sharpness:", measure_sharpness_laplacian(image))
    print(f"{name} MAE vs original:", measure_mae(gray_image, image))
    print(f"{name} MSE vs original:", measure_mse(gray_image, image))
    print(f"{name} PSNR vs original:", measure_psnr(gray_image, image))
    print()
    
print("Saved figure to:", output_path)