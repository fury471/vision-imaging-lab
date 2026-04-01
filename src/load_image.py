from image_utils import get_default_image_path, load_color_image


image_path = get_default_image_path()
image = load_color_image(image_path)

print("Image path:", image_path)
print("Shape:", image.shape)
print("Data type:", image.dtype)
