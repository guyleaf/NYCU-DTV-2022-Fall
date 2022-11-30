#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from loguru import logger

import cv2
import numpy as np
import matplotlib.pyplot as plt

from yolox.data.datasets.wrappers import Dataset
from yolox.evaluators.pascalvoc_2012.bounding_box import BoundingBox
from yolox.evaluators.pascalvoc_2012.bounding_boxes import BoundingBoxes
from yolox.evaluators.pascalvoc_2012.evaluator import Evaluator
from yolox.evaluators.pascalvoc_2012.utils import BBFormat, BBType
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

        logger.info("Loading GTA video dataset...")
        if image_set != "test":
            # load annotations
            anno_folder = os.path.join(data_dir, f"{image_set}_labels")
            self._ids, self._annos = self._load_annotations(anno_folder)
        else:
            self._annos = None
            self._ids = [
                int(filename.removesuffix(".jpg"))
                for filename in os.listdir(os.path.join(data_dir, image_set))
            ]
        logger.info(f"Number of {image_set} images: {len(self._ids)}")

    def _load_annotations(
        self, path: str
    ) -> tuple[list[int], list[np.ndarray]]:
        ids = []
        annos = []
        # load files in order by id
        for filename in os.listdir(path):
            # [class_ind, x_center, y_center, width, height]
            anno = pd.read_csv(
                os.path.join(path, filename),
                sep=" ",
                header=None,
            ).to_numpy()
            assert anno.shape[0] > 0

            anno = anno[:, [1, 2, 3, 4, 0]]

            # [x_center, y_center, width, height, class_ind]
            annos.append(anno)
            ids.append(int(filename.removesuffix(".txt")))
        return ids, annos

    def _load_image(self, id: int) -> cv2.Mat:
        img = cv2.imread(self._img_path % id, flags=cv2.IMREAD_COLOR)
        return img

    def _convert_to_voc_format(
        self, anno: np.ndarray, img_info: tuple[int, int]
    ) -> np.ndarray:
        anno[:, 0:4:2] *= img_info[1]
        anno[:, 1:4:2] *= img_info[0]
        anno = cxcywh2xyxy(anno, img_info).astype(dtype=np.float32)
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
            interpolation=cv2.INTER_LINEAR,
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

        if self._annos is not None:
            # avoid modifying original annotations
            anno = self._annos[index].copy()
            anno = self._convert_to_voc_format(anno, img_info)
        else:
            anno = np.zeros((0, 5), dtype=np.float32)

        img, anno = self._resize_image_and_annotation(img, anno)
        return img, anno, img_info, img_id


if __name__ == "__main__":
    from yolox.data import MosaicDataset
    from yolox.data.data_augment import TrainTransform, ValTransform

    dataset = GTAVideoDataset(
        "E:\\Git\\NYCU-DTV-2022-Fall\\hw2\\Code\\datasets\\GTA",
        image_set="val",
        image_size=(640, 640),
        preproc=TrainTransform(),
    )
    dataset = MosaicDataset(
        dataset,
        img_size=(640, 640),
        preproc=TrainTransform(max_labels=20, hsv_prob=0),
        mosaic=True,
    )
    logger.info("Checking data...")
    for data in dataset:
        wrapped_bboxes = BoundingBoxes()
        bboxes = data[1]
        bboxes = cxcywh2xyxy(data[1][:, [1, 2, 3, 4, 0]], data[2])
        bboxes = bboxes[np.sum(bboxes, axis=1) > 0]

        for bbox_type in [BBType.Detected, BBType.GroundTruth]:
            for x_min, y_min, x_max, y_max, class_id in bboxes:
                bbox = BoundingBox(
                    data[3],
                    class_id,
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                    bbType=bbox_type,
                    classConfidence=0.1,
                    format=BBFormat.XYX2Y2,
                )
                wrapped_bboxes.addBoundingBox(bbox)

        evaluator = Evaluator()
        results = evaluator.PlotPrecisionRecallCurve(wrapped_bboxes, 0.85)
        logger.info(results)

        img: np.ndarray = data[0].astype(dtype=np.uint8)
        img = np.transpose(img, axes=(1, 2, 0))
        img = np.ascontiguousarray(img)
        img = wrapped_bboxes.drawAllBoundingBoxes(img, data[3])
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()
        break
    logger.info(len(dataset))
