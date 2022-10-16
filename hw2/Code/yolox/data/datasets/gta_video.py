#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
from loguru import logger
from PIL import Image

import numpy as np
import pickle

from yolox.evaluators.voc_eval import voc_eval

from .wrappers import Dataset
from .gta_video_classes import GTA_CLASSES


DEFAULT_GTA_TRANSFORM = ""


class GTAVideoDataset(Dataset):

    """
    GTA Video Dataset Object

    input is image, target is annotation

    Args:
        root (string): filepath to VOCdevkit folder.
        image_set (string, optional): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
    """

    def __init__(
        self,
        data_dir: str,
        image_set: str = "train",
        transform=DEFAULT_GTA_TRANSFORM,
        cache: bool = False,
    ) -> None:
        self._root = data_dir
        self._img_set = image_set
        self._transform = transform
        self._img_path = os.path.join(data_dir, image_set, "%s.jpg")
        self._classes = GTA_CLASSES

        # load annotations
        anno_path = os.path.join(data_dir, f"{image_set}_labels", "%s.txt")
        self._annos = self._load_annotations(anno_path)

        # cache images or not
        self._imgs = None
        if cache:
            self._cache_images()

        super().__init__(self.img_size)

    def _load_annotations(self):
        return None

    def _load_image(self, index: int) -> Image:
        img = Image.open(self._img_path % index)
        return img

    def _cache_images(self):
        logger.warning(
            "\n********************************************************************************\n"
            "You are using cached images in RAM to accelerate training.\n"
            "This requires large system RAM.\n"
            "Make sure you have 60G+ RAM and 19G available disk space for training VOC.\n"
            "********************************************************************************\n"
        )
        max_h = self.img_size[0]
        max_w = self.img_size[1]
        cache_file = os.path.join(
            self.root, f"img_resized_cache_{self.name}.array"
        )
        if not os.path.exists(cache_file):
            logger.info(
                "Caching images for the first time. This might take about 3 minutes for VOC"
            )
            self.imgs = np.memmap(
                cache_file,
                shape=(len(self.ids), max_h, max_w, 3),
                dtype=np.uint8,
                mode="w+",
            )
            from tqdm import tqdm
            from multiprocessing.pool import ThreadPool

            NUM_THREADs = min(8, os.cpu_count())
            loaded_images = ThreadPool(NUM_THREADs).imap(
                lambda x: self.load_resized_img(x),
                range(len(self.annotations)),
            )
            pbar = tqdm(enumerate(loaded_images), total=len(self.annotations))
            for k, out in pbar:
                self.imgs[k][: out.shape[0], : out.shape[1], :] = out.copy()
            self.imgs.flush()
            pbar.close()
        else:
            logger.warning(
                "You are using cached imgs! Make sure your dataset is not changed!!\n"
                "Everytime the self.input_size is changed in your exp file, you need to delete\n"
                "the cached data and re-generate them.\n"
            )

        logger.info("Loading cached imgs...")
        self.imgs = np.memmap(
            cache_file,
            shape=(len(self.ids), max_h, max_w, 3),
            dtype=np.uint8,
            mode="r+",
        )

    @property
    def annotation(self, index: int):
        return self._annos[index][0]

    def pull_item(self, index: int) -> tuple:
        """Returns the original image and target at an index for mixup

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            img, target
        """
        if self.imgs is not None:
            target, img_info, resized_info = self.annotations[index]
            pad_img = self.imgs[index]
            img = pad_img[: resized_info[0], : resized_info[1], :].copy()
        else:
            img = self.load_resized_img(index)
            target, img_info, _ = self.annotations[index]

        return img, target, img_info, index

    def __len__(self) -> int:
        return len(self._annos)

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)

        return img, target, img_info, img_id

    # def evaluate_detections(self, all_boxes, output_dir=None):
    #     """
    #     all_boxes is a list of length number-of-classes.
    #     Each list element is a list of length number-of-images.
    #     Each of those list elements is either an empty list []
    #     or a numpy array of detection.

    #     all_boxes[class][image] = [] or np.array of shape #dets x 5
    #     """
    #     self._write_voc_results_file(all_boxes)
    #     IouTh = np.linspace(
    #         0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
    #     )
    #     mAPs = []
    #     for iou in IouTh:
    #         mAP = self._do_python_eval(output_dir, iou)
    #         mAPs.append(mAP)

    #     print("--------------------------------------------------------------")
    #     print("map_5095:", np.mean(mAPs))
    #     print("map_50:", mAPs[0])
    #     print("--------------------------------------------------------------")
    #     return np.mean(mAPs), mAPs[0]

    # def _get_voc_results_file_template(self):
    #     filename = "comp4_det_test" + "_{:s}.txt"
    #     filedir = os.path.join(
    #         self.root, "results", "VOC" + self._year, "Main"
    #     )
    #     if not os.path.exists(filedir):
    #         os.makedirs(filedir)
    #     path = os.path.join(filedir, filename)
    #     return path

    # def _write_voc_results_file(self, all_boxes):
    #     for cls_ind, cls in enumerate(GTA_CLASSES):
    #         cls_ind = cls_ind
    #         if cls == "__background__":
    #             continue
    #         print("Writing {} VOC results file".format(cls))
    #         filename = self._get_voc_results_file_template().format(cls)
    #         with open(filename, "wt") as f:
    #             for im_ind, index in enumerate(self.ids):
    #                 index = index[1]
    #                 dets = all_boxes[cls_ind][im_ind]
    #                 if dets == []:
    #                     continue
    #                 for k in range(dets.shape[0]):
    #                     f.write(
    #                         "{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n".format(
    #                             index,
    #                             dets[k, -1],
    #                             dets[k, 0] + 1,
    #                             dets[k, 1] + 1,
    #                             dets[k, 2] + 1,
    #                             dets[k, 3] + 1,
    #                         )
    #                     )

    # def _do_python_eval(self, output_dir="output", iou=0.5):
    #     rootpath = os.path.join(self.root, "VOC" + self._year)
    #     name = self.image_set[0][1]
    #     annopath = os.path.join(rootpath, "Annotations", "{:s}.xml")
    #     imagesetfile = os.path.join(
    #         rootpath, "ImageSets", "Main", name + ".txt"
    #     )
    #     cachedir = os.path.join(
    #         self.root, "annotations_cache", "VOC" + self._year, name
    #     )
    #     if not os.path.exists(cachedir):
    #         os.makedirs(cachedir)
    #     aps = []
    #     # The PASCAL VOC metric changed in 2010
    #     use_07_metric = True if int(self._year) < 2010 else False
    #     print("Eval IoU : {:.2f}".format(iou))
    #     if output_dir is not None and not os.path.isdir(output_dir):
    #         os.mkdir(output_dir)
    #     for i, cls in enumerate(GTA_CLASSES):

    #         if cls == "__background__":
    #             continue

    #         filename = self._get_voc_results_file_template().format(cls)
    #         rec, prec, ap = voc_eval(
    #             filename,
    #             annopath,
    #             imagesetfile,
    #             cls,
    #             cachedir,
    #             ovthresh=iou,
    #             use_07_metric=use_07_metric,
    #         )
    #         aps += [ap]
    #         if iou == 0.5:
    #             print("AP for {} = {:.4f}".format(cls, ap))
    #         if output_dir is not None:
    #             with open(
    #                 os.path.join(output_dir, cls + "_pr.pkl"), "wb"
    #             ) as f:
    #                 pickle.dump({"rec": rec, "prec": prec, "ap": ap}, f)
    #     if iou == 0.5:
    #         print("Mean AP = {:.4f}".format(np.mean(aps)))
    #         print("~~~~~~~~")
    #         print("Results:")
    #         for ap in aps:
    #             print("{:.3f}".format(ap))
    #         print("{:.3f}".format(np.mean(aps)))
    #         print("~~~~~~~~")
    #         print("")
    #         print(
    #             "--------------------------------------------------------------"
    #         )
    #         print("Results computed with the **unofficial** Python eval code.")
    #         print(
    #             "Results should be very close to the official MATLAB eval code."
    #         )
    #         print(
    #             "Recompute with `./tools/reval.py --matlab ...` for your paper."
    #         )
    #         print("-- Thanks, The Management")
    #         print(
    #             "--------------------------------------------------------------"
    #         )

    #     return np.mean(aps)
