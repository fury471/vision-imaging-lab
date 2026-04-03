import numpy as np
import cv2


def shift_brightness(gray_image, shift_value):
    gray_int = gray_image.astype(np.int16)
    shifted = np.clip(gray_int + shift_value, 0, 255).astype(np.uint8)
    return shifted


def scale_contrast(gray_image, alpha):
    return cv2.convertScaleAbs(gray_image, alpha=alpha, beta=0)


def scale_contrast_around_mean(gray_image, alpha):
    mean_intensity = np.mean(gray_image)
    gray_float = gray_image.astype(np.float32)

    centered = mean_intensity + alpha * (gray_float - mean_intensity)
    centered = np.clip(centered, 0, 255).astype(np.uint8)

    return centered

def histogram_equalization(gray_image):
    return cv2.equalizeHist(gray_image)


def apply_clahe(gray_image, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray_image)


def gaussian_denoise(gray_image, kernel_size=(5, 5), sigma=0):
    return cv2.GaussianBlur(gray_image, kernel_size, sigma)


def median_denoise(gray_image, kernel_size=5):
    return cv2.medianBlur(gray_image, kernel_size)


def bilateral_denoise(gray_image, d=9, sigma_color=50, sigma_space=50):
    return cv2.bilateralFilter(gray_image, d, sigma_color, sigma_space)

