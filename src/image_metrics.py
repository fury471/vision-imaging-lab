import numpy as np


def measure_brightness(gray_image):
    return np.mean(gray_image)


def measure_contrast(gray_image):
    return np.std(gray_image)


def measure_min_intensity(gray_image):
    return np.min(gray_image)


def measure_max_intensity(gray_image):
    return np.max(gray_image)
