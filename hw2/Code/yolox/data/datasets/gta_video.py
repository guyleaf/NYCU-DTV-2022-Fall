#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import copy
import os
from loguru import logger

import torch
from torch.utils.data import Dataset

import cv2
from PIL import Image

import numpy as np
import pandas as pd

import albumentations as A
from albumentations.pytorch import ToTensorV2

from yolox.data.datasets.gta_video_classes import GTA_CLASSES


DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)

DEFAULT_TRAIN_TRANSFORMS = A.Compose(
    [
        A.LongestMaxSize(416),
        A.Flip(p=0.5),
        A.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD),
        A.PadIfNeeded(
            min_height=416, min_width=416, border_mode=cv2.BORDER_CONSTANT
        ),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(
        format="yolo", min_visibility=0.2, label_fields=["class_labels"]
    ),
)

DEFAULT_VAL_TRANSFORMS = A.Compose(
    [
        A.LongestMaxSize(416),
        A.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD),
        A.PadIfNeeded(
            min_height=416, min_width=416, border_mode=cv2.BORDER_CONSTANT
        ),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
)


class GTAVideoDataset(Dataset):

    """
    GTA Video Dataset Object

    input is image, target is annotation

    Args:
        root (string): filepath to VOCdevkit folder.
        image_set (string, optional): imageset to use (eg. 'train', 'val')
        image_size (tuple[int, int], optional): (height, width)
        preproc (callable, optional): transformation to perform on the
            input image
    """

    def __init__(
        self,
        data_dir: str,
        image_set: str = "train",
        image_size: tuple[int, int] = (1080, 1920),
        preproc: A.Compose = DEFAULT_TRAIN_TRANSFORMS,
    ) -> None:
        super().__init__()
        self._root = data_dir
        self._img_set = image_set
        self._preproc = preproc
        self._img_path = os.path.join(data_dir, image_set, "%s.jpg")
        self._img_size = image_size
        self._classes = GTA_CLASSES

        logger.info("Loading GTA video dataset...")
        # load annotations
        anno_folder = os.path.join(data_dir, f"{image_set}_labels")
        self._ids, self._annos = self._load_annotations(anno_folder)
        logger.info(f"Number of {image_set} images:", len(self._ids))

    def _load_annotations(self, path: str) -> list:
        ids = []
        annos = []
        # load files in order by id
        for filename in os.listdir(path):
            # [class_ind, x_center, y_center, width, height]
            content = pd.read_csv(os.path.join(path, filename), sep=" ")

            bboxes = content.iloc[:, 1:].values.tolist()
            class_inds = content.iloc[:, 0].values.tolist()

            # [x_center, y_center, width, height, class_ind]
            annos.append([bboxes, class_inds])
            ids.append(filename.removesuffix(".txt"))
        return ids, annos

    def _load_image(self, index: int) -> np.ndarray:
        img_id = self._ids[index]
        img = Image.open(self._img_path % img_id)
        return np.array(img)

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, index):
        img, (bboxes, class_inds), img_info, img_id = self.pull_item(index)

        if self._preproc is not None:
            transformed_data = self._preproc(
                image=img, bboxes=bboxes, class_labels=class_inds
            )
            img = transformed_data["image"]
            bboxes = transformed_data["bboxes"]
            class_inds = transformed_data["class_labels"]

        target = [
            torch.tensor(bboxes, dtype=torch.float32),
            torch.tensor(class_inds),
        ]
        return img, target, img_info, img_id

    def pull_item(self, index: int) -> tuple:
        """Returns the original image and target at an index for mixup

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            img, target
        """
        img = self._load_image(index)
        target = self._annos[index]

        return img, target, copy.deepcopy(self._img_size), index


if __name__ == "__main__":
    dataset = GTAVideoDataset(
        "E:\\Git\\NYCU-DTV-2022-Fall\\hw2\\Code\\datasets\\GTA",
        image_set="val",
    )
    print("Checking data...")
    for data in dataset:
        pass
    print(len(dataset))
