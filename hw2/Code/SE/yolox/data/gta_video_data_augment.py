from loguru import logger
import cv2
import numpy as np
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2
from yolox.utils.boxes import xyxy2cxcywh


class TrainTransform:
    def __init__(
        self,  # [height, width]
        max_labels: int = 50,
        image_size: tuple[int, int] = (416, 416),
        jitter_prob: float = 0.5,
        flip_prob: float = 0.5,
        jitter_brightness: float = 0.2,
        jitter_contrast: float = 0.2,
        jitter_saturation: float = 0.2,
        jitter_hue: float = 0.2,
        padding_value: int = 0,
        bbox_min_area: float = 30,
        bbox_min_visibility: float = 0.2,
    ) -> None:
        self._max_labels = max_labels
        self._transform = A.Compose(
            [
                A.LongestMaxSize(
                    max_size=max(image_size), interpolation=cv2.INTER_AREA
                ),
                A.ColorJitter(
                    brightness=jitter_brightness,
                    contrast=jitter_contrast,
                    saturation=jitter_saturation,
                    hue=jitter_hue,
                    p=jitter_prob,
                ),
                A.HorizontalFlip(p=flip_prob),
                A.PadIfNeeded(
                    min_height=image_size[0],
                    min_width=image_size[1],
                    border_mode=cv2.BORDER_CONSTANT,
                    value=padding_value,
                ),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                min_visibility=bbox_min_visibility,
                min_area=bbox_min_area,
            ),
        )

    def _filter_target(self, target: np.ndarray):
        x_diff = target[:, 2] - target[:, 0]
        y_diff = target[:, 3] - target[:, 1]
        return target[x_diff * y_diff > 0]

    def __call__(
        self, img: np.ndarray, target: np.ndarray, _: tuple[int, int]
    ) -> tuple[torch.Tensor, np.ndarray, list]:
        target = self._filter_target(target)
        transformed_data = self._transform(image=img, bboxes=target)
        img = transformed_data["image"].float()
        target = np.array(transformed_data["bboxes"], dtype=np.float32)

        if target.shape[0] == 0:
            target = np.zeros((self._max_labels, 5), dtype=np.float32)
            return img, target

        target = xyxy2cxcywh(target)

        mask_b = np.minimum(target[:, 2], target[:, 3]) > 1
        masked_target = target[mask_b]
        if masked_target.shape[0] == 0:
            masked_target = target

        masked_target = masked_target[:, [4, 0, 1, 2, 3]]
        diff = self._max_labels - masked_target.shape[0]
        if diff > 0:
            masked_target = np.pad(
                masked_target,
                pad_width=((0, diff), (0, 0)),
                constant_values=0,
            )
        else:
            masked_target = masked_target[: self._max_labels]
        return img, masked_target


class ValTransform:
    def __init__(
        self,  # [height, width]
        image_size: tuple[int, int] = (416, 416),
        padding_value: int = 0,
    ) -> None:
        self._transform = A.Compose(
            [
                A.LongestMaxSize(
                    max_size=max(image_size), interpolation=cv2.INTER_AREA
                ),
                A.PadIfNeeded(
                    min_height=image_size[0],
                    min_width=image_size[1],
                    border_mode=cv2.BORDER_CONSTANT,
                    value=padding_value,
                ),
                ToTensorV2(),
            ]
        )

    def __call__(
        self, img: np.ndarray, target: np.ndarray, _: tuple[int, int]
    ) -> tuple[torch.Tensor, np.ndarray, list]:
        transformed_data = self._transform(image=img)
        img = transformed_data["image"].float()
        return img, np.zeros((1, 5), dtype=np.float32)
