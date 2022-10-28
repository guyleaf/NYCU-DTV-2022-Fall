#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from loguru import logger

import cv2
import numpy as np
import matplotlib.pyplot as plt

from yolox.data.datasets.gta_video_classes import GTA_CLASSES
from yolox.data.datasets.wrappers import Dataset, MosaicDataset
from yolox.utils.boxes import cxcywh2xyxy

import os
import pandas as pd


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
        image_size: tuple[int, int] = (416, 416),
        preproc=None,
    ) -> None:
        super().__init__(image_size)
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
        logger.info(f"Number of {image_set} images: {len(self._ids)}")

    def _load_annotations(
        self, path: str
    ) -> tuple[list[str], list[np.ndarray]]:
        ids = []
        annos = []
        # load files in order by id
        for filename in os.listdir(path):
            # [class_ind, x_center, y_center, width, height]
            bboxes = pd.read_csv(
                os.path.join(path, filename), sep=" "
            ).to_numpy()

            bboxes = bboxes[:, [1, 2, 3, 4, 0]]

            # [x_center, y_center, width, height, class_ind]
            annos.append(bboxes)
            ids.append(filename.removesuffix(".txt"))
        return ids, annos

    def _load_image(self, id: int) -> cv2.Mat:
        img = cv2.imread(self._img_path % id, flags=cv2.IMREAD_COLOR)
        return img

    def _convert_to_voc_format(
        self, anno: np.ndarray, img_info: tuple[int, int]
    ) -> np.ndarray:
        anno[:, 0:4:2] *= img_info[1]
        anno[:, 1:4:2] *= img_info[0]
        anno = cxcywh2xyxy(anno, img_info)
        return anno

    def _resize_image_and_annotation(
        self, img: cv2.Mat, anno: np.ndarray
    ) -> tuple[cv2.Mat, np.ndarray]:
        r = min(
            self._img_size[0] / img.shape[0], self._img_size[1] / img.shape[1]
        )
        img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_AREA,
        ).astype(dtype=np.uint8)
        anno[:, :4] *= r
        return img, anno

    def __len__(self) -> int:
        return len(self._ids)

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        img, target, img_info, img_id = self.pull_item(index)

        if self._preproc is not None:
            img, target = self._preproc(img, target, self.input_dim)

        return img, target, img_info, img_id

    def load_anno(self, index: int) -> tuple[float, ...]:
        return self._annos[index][0]

    def pull_item(self, index: int) -> tuple[np.ndarray, list, dict, int]:
        """Returns the original image and target at an index for mixup

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            img, target
        """
        img_id = self._ids[index]
        img = self._load_image(img_id)
        img_info = tuple(img.shape[:2])
        anno = self._annos[index]

        anno = self._convert_to_voc_format(anno, img_info)
        # img, anno = self._resize_image_and_annotation(img, anno)
        return img, anno, img_info, img_id


if __name__ == "__main__":
    from yolox.data.gta_video_data_augment import TrainTransform

    dataset = GTAVideoDataset(
        "E:\\Git\\NYCU-DTV-2022-Fall\\hw2\\Code\\datasets\\GTA",
        image_set="val",
        preproc=TrainTransform(),
    )
    dataset = MosaicDataset(
        dataset, img_size=(416, 416), preproc=TrainTransform(jitter_prob=0.0), mosaic=False
    )
    logger.info("Checking data...")
    for data in dataset:
        logger.info(data[0].dtype)
        logger.info(data[1])
        logger.info(data[3])
        img: np.ndarray = data[0].numpy().astype(dtype=np.uint8)
        img = np.transpose(img, axes=(1, 2, 0))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()
        break
    logger.info(len(dataset))
