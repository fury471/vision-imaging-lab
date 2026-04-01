from pathlib import Path

import cv2


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_default_image_path() -> Path:
    return get_project_root() / "data" / "raw" / "test.jpg"


def load_color_image(image_path: Path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    return image


def to_grayscale(image_bgr):
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
