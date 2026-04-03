import numpy as np
import matplotlib.pyplot as plt

from image_metrics import (
    estimate_noise_std,
    measure_psnr,
    measure_sharpness_laplacian,
)
from image_transforms import bilateral_denoise
from image_utils import get_default_image_path, get_project_root, load_color_image, to_grayscale


image_path = get_default_image_path()
output_path = get_project_root() / "outputs" / "bilateral_tuning.png"

image_bgr = load_color_image(image_path)
gray_image = to_grayscale(image_bgr)

np.random.seed(42)
noise = np.random.normal(0, 15, gray_image.shape)
noisy_image = np.clip(gray_image.astype("float32") + noise, 0, 255).astype("uint8")

bilateral_soft = bilateral_denoise(noisy_image, d=7, sigma_color=30, sigma_space=30)
bilateral_default = bilateral_denoise(noisy_image, d=9, sigma_color=50, sigma_space=50)
bilateral_strong = bilateral_denoise(noisy_image, d=11, sigma_color=80, sigma_space=80)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.subplots_adjust(hspace=0.25, wspace=0.05)

axes[0, 0].imshow(noisy_image, cmap="gray")
axes[0, 0].set_title("Noisy")
axes[0, 0].axis("off")

axes[0, 1].imshow(bilateral_soft, cmap="gray")
axes[0, 1].set_title("Bilateral Soft")
axes[0, 1].axis("off")

axes[1, 0].imshow(bilateral_default, cmap="gray")
axes[1, 0].set_title("Bilateral Default")
axes[1, 0].axis("off")

axes[1, 1].imshow(bilateral_strong, cmap="gray")
axes[1, 1].set_title("Bilateral Strong")
axes[1, 1].axis("off")

plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()

for name, image in [
    ("Noisy", noisy_image),
    ("Soft", bilateral_soft),
    ("Default", bilateral_default),
    ("Strong", bilateral_strong),
]:
    print(f"{name} estimated noise:", estimate_noise_std(image))
    print(f"{name} sharpness:", measure_sharpness_laplacian(image))
    print(f"{name} PSNR vs original:", measure_psnr(gray_image, image))
    print()
    
print("Saved figure to:", output_path)