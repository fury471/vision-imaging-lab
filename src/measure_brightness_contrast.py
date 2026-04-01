from image_metrics import (
    measure_brightness,
    measure_contrast,
    measure_max_intensity,
    measure_min_intensity,
)
from image_utils import get_default_image_path, load_color_image, to_grayscale


image_path = get_default_image_path()
image_bgr = load_color_image(image_path)
gray_image = to_grayscale(image_bgr)

brightness = measure_brightness(gray_image)
contrast = measure_contrast(gray_image)
min_intensity = measure_min_intensity(gray_image)
max_intensity = measure_max_intensity(gray_image)

print("Image path:", image_path)
print("Brightness (mean intensity):", brightness)
print("Contrast (std intensity):", contrast)
print("Min intensity:", min_intensity)
print("Max intensity:", max_intensity)
