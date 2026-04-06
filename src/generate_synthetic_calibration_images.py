from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_DIR = PROJECT_ROOT / "data" / "calibration"
REFERENCE_OUTPUT = PROJECT_ROOT / "outputs" / "calibration_board_reference.png"

BOARD_SQUARES = (9, 6)
SQUARE_SIZE = 80
BORDER = 80
CANVAS_SIZE = (1280, 960)  # width, height


def create_chessboard_board() -> np.ndarray:
    cols, rows = BOARD_SQUARES
    board_width = cols * SQUARE_SIZE
    board_height = rows * SQUARE_SIZE

    board = np.full(
        (board_height + 2 * BORDER, board_width + 2 * BORDER, 3),
        255,
        dtype=np.uint8,
    )

    for row in range(rows):
        for col in range(cols):
            color = 0 if (row + col) % 2 == 0 else 255
            x0 = BORDER + col * SQUARE_SIZE
            y0 = BORDER + row * SQUARE_SIZE
            x1 = x0 + SQUARE_SIZE
            y1 = y0 + SQUARE_SIZE
            board[y0:y1, x0:x1] = color

    return board


def create_background(width: int, height: int) -> np.ndarray:
    x = np.linspace(205, 235, width, dtype=np.float32)
    y = np.linspace(225, 245, height, dtype=np.float32).reshape(-1, 1)
    gradient = np.clip((x + y) / 2.0, 0, 255).astype(np.uint8)
    background = np.stack([gradient, gradient, gradient], axis=2)
    return background


def apply_board_to_canvas(board: np.ndarray, destination_points: np.ndarray, index: int) -> np.ndarray:
    width, height = CANVAS_SIZE
    canvas = create_background(width, height)

    src_points = np.array(
        [
            [0, 0],
            [board.shape[1] - 1, 0],
            [board.shape[1] - 1, board.shape[0] - 1],
            [0, board.shape[0] - 1],
        ],
        dtype=np.float32,
    )

    transform = cv2.getPerspectiveTransform(src_points, destination_points.astype(np.float32))
    warped = cv2.warpPerspective(board, transform, (width, height))
    mask = cv2.warpPerspective(
        np.full(board.shape[:2], 255, dtype=np.uint8),
        transform,
        (width, height),
    )

    canvas[mask > 0] = warped[mask > 0]

    if index % 3 == 0:
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

    if index % 4 == 0:
        rng = np.random.default_rng(index)
        noise = rng.normal(0, 2.5, canvas.shape).astype(np.float32)
        canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return canvas


def main() -> None:
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    board = create_chessboard_board()
    cv2.imwrite(str(REFERENCE_OUTPUT), board)

    destination_sets = [
        np.array([[240, 170], [980, 140], [1030, 700], [210, 730]], dtype=np.float32),
        np.array([[310, 120], [930, 180], [940, 760], [260, 700]], dtype=np.float32),
        np.array([[180, 240], [900, 130], [1060, 620], [320, 760]], dtype=np.float32),
        np.array([[360, 180], [1030, 230], [900, 770], [190, 690]], dtype=np.float32),
        np.array([[220, 110], [870, 150], [980, 680], [150, 640]], dtype=np.float32),
        np.array([[410, 150], [980, 120], [1040, 640], [330, 760]], dtype=np.float32),
        np.array([[230, 200], [1040, 170], [920, 760], [160, 690]], dtype=np.float32),
        np.array([[290, 150], [930, 250], [1010, 700], [250, 650]], dtype=np.float32),
        np.array([[180, 160], [860, 120], [1080, 720], [270, 790]], dtype=np.float32),
        np.array([[350, 230], [1030, 200], [960, 760], [280, 670]], dtype=np.float32),
        np.array([[260, 120], [980, 150], [1080, 710], [170, 650]], dtype=np.float32),
        np.array([[380, 130], [920, 170], [980, 790], [340, 730]], dtype=np.float32),
    ]

    for idx, destination_points in enumerate(destination_sets, start=1):
        image = apply_board_to_canvas(board, destination_points, idx)
        image_path = CALIBRATION_DIR / f"calibration_{idx:02d}.png"
        cv2.imwrite(str(image_path), image)

    print(f"Saved {len(destination_sets)} synthetic calibration images to: {CALIBRATION_DIR}")
    print(f"Saved board reference image to: {REFERENCE_OUTPUT}")
    print("Board squares: 9 x 6")
    print("Inner corners for OpenCV: 8 x 5")


if __name__ == "__main__":
    main()
