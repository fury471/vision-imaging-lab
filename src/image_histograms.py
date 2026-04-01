import numpy as np


def compute_grayscale_histogram(gray_image, num_bins=256):
    histogram, bin_edges = np.histogram(gray_image, bins=num_bins, range=(0, 256))
    return histogram, bin_edges


def count_pixels(gray_image):
    return gray_image.shape[0] * gray_image.shape[1]
