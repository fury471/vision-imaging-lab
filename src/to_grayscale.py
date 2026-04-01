from pathlib import Path

import cv2

from image_utils import get_default_image_path, get_project_root, load_color_image, to_grayscale


image_path = get_default_image_path()
output_path = get_project_root() / "outputs" / "test_gray.jpg"

image_bgr = load_color_image(image_path)
gray_image = to_grayscale(image_bgr)

cv2.imwrite(str(output_path), gray_image)

print("Image path:", image_path)
print("Output path:", output_path)
print("Original shape:", image_bgr.shape)
print("Gray shape:", gray_image.shape)
print("Original dtype:", image_bgr.dtype)
print("Gray dtype:", gray_image.dtype)
print("Top-left BGR pixel:", image_bgr[0, 0])
print("Top-left gray pixel:", gray_image[0, 0])
