from loguru import logger
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import tempfile
from tabulate import tabulate

from .pascalvoc_2012.bounding_box import BoundingBox
from .pascalvoc_2012.bounding_boxes import BoundingBoxes
from .pascalvoc_2012.evaluator import Evaluator as PascalVOC2012Evaluator
from .pascalvoc_2012.utils import BBFormat


class GTAVideoEvaluator:
    """
    GTA Video Evaluation class.
    Use PASCAL VOC 2012 metrics implemented by https://github.com/rafaelpadilla/Object-Detection-Metrics
    """

    def __init__(
        self,
        dataloader: DataLoader,
        conf_threshold: float,
        iou_threshold: float,
        num_classes: int,
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            conf_threshold (float): confidence threshold ranging from 0 to 1,
                                    which is defined in the config file.
            iou_threshold (float): IOU threshold ranging from 0 to 1
                                    which is defined in the config file
                                    indicates which detections will be considered TP or FP.
        """
        self._dataloader = dataloader
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold
        self._num_classes = num_classes
        self._num_images = len(dataloader.dataset)

    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
    ):
        pass
