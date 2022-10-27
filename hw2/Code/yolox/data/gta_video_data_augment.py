import cv2
import numpy as np
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2


class TrainTransform:
    def __init__(
        self,  # [height, width]
        input_size: tuple[int, int] = (416, 416),
        jitter_prob: float = 0.5,
        flip_prob: float = 0.5,
        jitter_brightness: float = 0.2,
        jitter_contrast: float = 0.2,
        jitter_saturation: float = 0.2,
        jitter_hue: float = 0.2,
        padding_value: int = 0,
        bbox_min_visibility: float = 0.2,
    ) -> None:
        self._transform = A.Compose(
            [
                A.ColorJitter(
                    brightness=jitter_brightness,
                    contrast=jitter_contrast,
                    saturation=jitter_saturation,
                    hue=jitter_hue,
                    p=jitter_prob,
                ),
                A.HorizontalFlip(p=flip_prob),
                A.PadIfNeeded(
                    min_height=input_size[0],
                    min_width=input_size[1],
                    border_mode=cv2.BORDER_CONSTANT,
                    value=padding_value,
                ),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc", min_visibility=bbox_min_visibility
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
        target = target[:, [4, 0, 1, 2, 3]]
        return img, target


class ValTransform:
    def __init__(
        self,  # [height, width]
        input_size: tuple[int, int] = (416, 416),
        padding_value: int = 0,
    ) -> None:
        self._transform = A.Compose(
            [
                A.PadIfNeeded(
                    min_height=input_size[0],
                    min_width=input_size[1],
                    border_mode=cv2.BORDER_CONSTANT,
                    value=padding_value,
                ),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc"),
        )

    def __call__(
        self, img: np.ndarray, target: np.ndarray, _: tuple[int, int]
    ) -> tuple[torch.Tensor, np.ndarray, list]:
        transformed_data = self._transform(image=img, bboxes=target)
        img = transformed_data["image"].float()
        target = np.array(transformed_data["bboxes"], dtype=np.float32)
        target = target[:, [4, 0, 1, 2, 3]]
        return img, target
