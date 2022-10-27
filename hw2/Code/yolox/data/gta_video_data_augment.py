import cv2
import numpy as np
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2

DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


class TrainTransform:
    def __init__(
        self,  # [height, width]
        input_size: tuple[int, int] = (416, 416),
        mean: tuple[float, float, float] = DEFAULT_MEAN,
        std: tuple[float, float, float] = DEFAULT_STD,
    ) -> None:
        self._transform = A.Compose(
            [
                A.LongestMaxSize(max(input_size)),
                A.Flip(p=0.5),
                A.Normalize(mean=mean, std=std),
                A.PadIfNeeded(
                    min_height=input_size[0],
                    min_width=input_size[1],
                    border_mode=cv2.BORDER_CONSTANT,
                ),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="yolo",
                min_visibility=0.2,
                label_fields=["class_labels"],
            ),
        )

    def __call__(
        self, image: np.ndarray, bboxes: list, labels: list
    ) -> tuple[torch.Tensor, list, list]:
        transformed_data = self._transform(
            image=image, bboxes=bboxes, class_labels=labels
        )
        return (
            transformed_data["image"],
            transformed_data["bboxes"],
            transformed_data["class_labels"],
        )


class ValTransform(TrainTransform):
    def __init__(
        self,  # [height, width]
        input_size: tuple[int, int] = (416, 416),
        mean: tuple[float, float, float] = DEFAULT_MEAN,
        std: tuple[float, float, float] = DEFAULT_STD,
    ) -> None:
        super().__init__(input_size, mean, std)
        self._transform = A.Compose(
            [
                A.LongestMaxSize(max(input_size)),
                A.Normalize(mean=mean, std=std),
                A.PadIfNeeded(
                    min_height=input_size[0],
                    min_width=input_size[1],
                    border_mode=cv2.BORDER_CONSTANT,
                ),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="yolo",
                min_visibility=0.2,
                label_fields=["class_labels"],
            ),
        )


def get_train_transforms(
    # [height, width]
    input_size: tuple[int, int] = (416, 416),
    mean: tuple[float, float, float] = DEFAULT_MEAN,
    std: tuple[float, float, float] = DEFAULT_STD,
):
    trasnforms = A.Compose(
        [
            A.LongestMaxSize(max(input_size)),
            A.Flip(p=0.5),
            A.Normalize(mean=mean, std=std),
            A.PadIfNeeded(
                min_height=input_size[0],
                min_width=input_size[1],
                border_mode=cv2.BORDER_CONSTANT,
            ),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="yolo", min_visibility=0.2, label_fields=["class_labels"]
        ),
    )

    def transform(image: np.ndarray, bboxes: list, labels: list):
        transformed_data = trasnforms(
            image=image, bboxes=bboxes, class_labels=labels
        )
        return (
            transformed_data["image"],
            transformed_data["bboxes"],
            transformed_data["class_labels"],
        )

    return transform


def get_val_transforms(
    # [height, width]
    input_size: tuple[int, int] = (416, 416),
    mean: tuple[float, float, float] = DEFAULT_MEAN,
    std: tuple[float, float, float] = DEFAULT_STD,
):
    trasnforms = A.Compose(
        [
            A.LongestMaxSize(max(input_size)),
            A.Normalize(mean=mean, std=std),
            A.PadIfNeeded(
                min_height=input_size[0],
                min_width=input_size[1],
                border_mode=cv2.BORDER_CONSTANT,
            ),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )

    def transform(image: np.ndarray, bboxes: list, labels: list):
        transformed_data = trasnforms(
            image=image, bboxes=bboxes, class_labels=labels
        )
        return (
            transformed_data["image"],
            transformed_data["bboxes"],
            transformed_data["class_labels"],
        )

    return transform
