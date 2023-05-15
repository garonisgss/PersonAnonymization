import cv2
import os
import detectron2
import numpy as np

from typing import Tuple, List


def filter_with_lines(crop: Tuple[int, int, int, int], lines: List[Tuple[float, float, bool]]) -> bool:
    x1, x2, y1, y2 = crop
    x_mid = (x1 + x2) / 2
    y_mid = (y1 + y2) / 2

    return all((a + b * x_mid > y_mid) == under for a, b, under in lines)


def cut_predicted_people(
    img: np.ndarray,
    preds: detectron2.structures.Instances,
    name_prefix: str,
    output_path: str,
    delta: float = 0.2,
    selected_class: int = 0,
    lines: List[Tuple[float, float, bool]] = None,
    min_width: int = 35,
    min_height: int = 35,
):
    if lines is None:
        lines = []

    for i, pic in enumerate(preds[preds.pred_classes == selected_class].pred_boxes):
        pic = pic.cpu().numpy()

        x1, y1, x2, y2 = pic
        x_diff = x2 - x1
        y_diff = y2 - y1
        x1 = int(np.floor(x1 - x_diff * delta / 2))
        x2 = int(np.ceil(x2 + x_diff * delta / 2))
        y1 = int(np.floor(y1 - y_diff * delta / 2))
        y2 = int(np.ceil(y2 + y_diff * delta / 2))

        im_box = img[y1:y2, x1:x2]

        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        if (
            im_box.size > 0
            and filter_with_lines((x1, x2, y1, y2), lines)
            and x_diff >= min_width
            and y_diff >= min_height
        ):
            cv2.imwrite(f"{output_path}/{name_prefix}_{i}.jpg", im_box)
