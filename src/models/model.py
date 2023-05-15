from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


def get_detectron_model(device: str = "cpu", threshold: float = 0.25) -> DefaultPredictor:
    if threshold < 1e-5 or threshold > 1 - 1e-5:
        raise AttributeError("Threshold should be a float number in (0, 1)")

    cfg = get_cfg()
    cfg.MODEL.DEVICE = device
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    return DefaultPredictor(cfg)
