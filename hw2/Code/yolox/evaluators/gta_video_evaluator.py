from loguru import logger

import tempfile
import torch


from tabulate import tabulate
from torch.utils.data import DataLoader
from tqdm import tqdm


class GTAVideoEvaluator:
    """
    GTA Video Evaluation class.
    Use PASCAL VOC 2012 metrics implemented by https://github.com/rafaelpadilla/Object-Detection-Metrics
    """

    def __init__(
        self,
        dataloader: DataLoader,
        conf_threshold: float,
        nms_threshold: float,
        num_classes: int,
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            conf_threshold (float): confidence threshold ranging from 0 to 1,
                                    which is defined in the config file.
            nms_threshold (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self._dataloader = dataloader
        self._conf_threshold = conf_threshold
        self._nms_threshold = nms_threshold
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
