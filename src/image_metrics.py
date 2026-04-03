import numpy as np
import cv2


def measure_brightness(gray_image):
    return np.mean(gray_image)


def measure_contrast(gray_image):
    return np.std(gray_image)


def measure_min_intensity(gray_image):
    return np.min(gray_image)


def measure_max_intensity(gray_image):
    return np.max(gray_image)


def measure_sharpness_laplacian(gray_image):
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    return laplacian.var()


def estimate_noise_std(gray_image):
    smoothed = cv2.GaussianBlur(gray_image, (5, 5), 0)
    residual = gray_image.astype("float32") - smoothed.astype("float32")
    return residual.std()


def measure_mae(reference_image, test_image):
    diff = reference_image.astype("float32") - test_image.astype("float32")
    return np.mean(np.abs(diff))


def measure_mse(reference_image, test_image):
    diff = reference_image.astype("float32") - test_image.astype("float32")
    return np.mean(diff ** 2)


def measure_psnr(reference_image, test_image):
    mse = measure_mse(reference_image, test_image)

    if mse == 0:
        return float("inf")

    max_pixel = 255.0
    return 10 * np.log10((max_pixel ** 2) / mse)

