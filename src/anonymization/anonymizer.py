import cv2
import numpy as np
import os

from tqdm.auto import tqdm

from src.models.model import get_detectron_model


def get_mask(img: np.ndarray, model, detectron_selected_class: int = 0):
    preds = model(img)["instances"]
    if np.any(preds.pred_classes.cpu().numpy() == detectron_selected_class):
        return np.any(
            preds[preds.pred_classes == detectron_selected_class].pred_masks[:, :, :].cpu().numpy(), axis=0
        )
    else:
        return np.full((preds.pred_masks.shape[1], preds.pred_masks.shape[2]), False)


def anonymize(
    img: np.ndarray,
    mask: np.ndarray = None,
    blur_mode: str = "median",
    blur_kernel_size: int = 11,
    colored: bool = True,
) -> np.ndarray:
    if not colored:
        img_anon = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_anon = img.copy()

    if blur_mode == "median":
        img_anon = cv2.medianBlur(img_anon, blur_kernel_size)
    elif blur_mode == "gaussian":
        img_anon = cv2.GaussianBlur(img_anon, (blur_kernel_size, blur_kernel_size), 0)
    elif blur_mode == "color_red":
        img_anon = np.zeros_like(img_anon)
        img_anon[:, :, 0] = 255
    else:
        raise AttributeError("Wrong blur_mode value")

    if mask is not None:
        if not colored:
            tmp = img_anon
            img_anon = img.copy()
            img_anon[mask, :] = tmp[mask, np.newaxis]
        else:
            img_anon[~mask, :] = img[~mask, :]

    return img_anon


def run_pipeline(
    input_path: str,
    output_folder_path: str,
    device: str = "cpu",
    detectron_threshold: float = 0.25,
    blur_mode: str = "median",
    blur_kernel_size: int = 11,
    colored: bool = True,
    detectron_selected_class: int = 0,
    enable_anonymization: bool = True,
    detect_blur_kernel_size: bool = True
):
    model = get_detectron_model(device, detectron_threshold)

    for root, dirs, files in tqdm(os.walk(input_path), "Iterate over folders"):
        base_dir = os.path.join(output_folder_path, os.path.relpath(root, input_path))
        for f in tqdm(files, "Iterate over pictures", leave=False):
            file_path = os.path.join(root, f)
            img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_RGB2BGR)

            preds = model(img)["instances"]

            if np.any(preds.pred_classes.cpu().numpy() == detectron_selected_class):
                mask = np.any(
                    preds[preds.pred_classes == detectron_selected_class].pred_masks[:, :, :].cpu().numpy(), axis=0
                )
                if detect_blur_kernel_size:
                    blur_kernel_size = int(max(img.shape[0], img.shape[1]) / 35)
                    blur_kernel_size = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
                if enable_anonymization:
                    img = anonymize(img, mask, blur_mode, blur_kernel_size, colored)

                prefix = f[: f.rfind(".")]
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                os.makedirs(base_dir, exist_ok=True)
                cv2.imwrite(f"{base_dir}/{prefix}.jpg", img)
