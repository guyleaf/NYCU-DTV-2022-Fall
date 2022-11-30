#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from yolox.data import get_yolox_datadir
from yolox.evaluators.voc2012_evaluator import VOC2012Evaluator
from yolox.exp import Exp as MyExp

import os


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        # ---------------- model config ---------------- #
        # detect classes number of model
        self.num_classes = 1
        # factor of model depth
        self.depth = 0.33
        # factor of model width
        self.width = 0.50
        # activation name. For example, if using "relu", then "silu" will be replaced to "relu".
        self.act = "silu"
        # iou loss. iou, giou
        self.iou_loss_type = "iou"
        self.depthwise = False
        self.sequeeze_backbone = True
        self.sequeeze_fpn = True

        self.max_epoch = 200
        self.no_aug_epochs = 20
        self.max_labels = 50
        self.data_dir = os.path.join(get_yolox_datadir(), "GTA")

        # --------------- transform config ----------------- #
        # prob of applying mosaic aug
        self.mosaic_prob = 1.0
        # prob of applying mixup aug
        self.mixup_prob = 1.0
        # prob of applying hsv aug
        self.hsv_prob = 1.0
        # prob of applying flip aug
        self.flip_prob = 0.5
        # rotation angle range, for example, if set to 2, the true range is (-2, 2)
        self.degrees = 10.0
        # translate range, for example, if set to 0.1, the true range is (-0.1, 0.1)
        self.translate = 0.1
        self.mosaic_scale = (0.1, 2)
        # apply mixup aug or not
        self.enable_mixup = True
        self.mixup_scale = (0.5, 1.5)
        # shear angle range, for example, if set to 2, the true range is (-2, 2)
        self.shear = 2.0

        # --------------  training config --------------------- #
        self.seed = 1234
        # eval period in epoch, for example,
        # if set to 1, model will be evaluate after every epoch.
        self.eval_interval = 10
        # save history checkpoint or not.
        # If set to False, yolox will only save latest and best ckpt.
        self.save_history_ckpt = True
        # name of experiment
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(
            "."
        )[0]

    def get_model(self):
        from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(
                self.depth,
                self.width,
                in_channels=in_channels,
                act=self.act,
                depthwise=self.depthwise,
                sequeeze_backbone=self.sequeeze_backbone,
                sequeeze_fpn=self.sequeeze_fpn,
            )
            head = YOLOXHead(
                self.num_classes,
                self.width,
                in_channels=in_channels,
                act=self.act,
                iou_loss_type=self.iou_loss_type,
                depthwise=self.depthwise,
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model

    def get_data_loader(
        self,
        batch_size: int,
        is_distributed: bool,
        no_aug: bool = False,
        cache_img: bool = False,
    ):
        from yolox.data import (
            DataLoader,
            GTAVideoDataset,
            InfiniteSampler,
            MosaicDataset,
            TrainTransform,
            YoloBatchSampler,
            worker_init_reset_seed,
        )
        from yolox.utils import wait_for_the_master

        with wait_for_the_master():
            dataset = GTAVideoDataset(
                data_dir=self.data_dir,
                image_set="train",
                image_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=self.max_labels,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob,
                ),
            )

        self.dataset = MosaicDataset(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=self.max_labels * 2 + 20,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
            ),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "batch_sampler": batch_sampler,
            # Make sure each process has different random seed, especially for 'fork' method
            "worker_init_fn": worker_init_reset_seed,
        }
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(
        self, batch_size, is_distributed, eval_set="val", legacy=False
    ):
        from yolox.data import GTAVideoDataset, ValTransform

        dataset = GTAVideoDataset(
            data_dir=self.data_dir,
            image_set=eval_set,
            image_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = DistributedSampler(dataset, shuffle=False)
        else:
            sampler = SequentialSampler(dataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
            "batch_size": batch_size,
        }
        val_loader = TorchDataLoader(dataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(
        self, batch_size, is_distributed, testdev=False, legacy=False
    ):
        from yolox.evaluators import VOC2012Evaluator

        train_loader = self.get_eval_loader(
            batch_size, is_distributed, eval_set="train", legacy=legacy
        )
        val_loader = self.get_eval_loader(
            batch_size,
            is_distributed,
            eval_set="val",
            legacy=legacy,
        )

        evaluator = VOC2012Evaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            conf_threshold=self.test_conf,
            nms_threshold=self.nmsthre,
            num_classes=self.num_classes,
            train_dataloader=train_loader,
        )
        return evaluator

    def eval(
        self,
        model,
        evaluator: VOC2012Evaluator,
        is_distributed: bool,
        half=False,
    ):
        return evaluator.evaluate(model, is_distributed, half, eval_train=True)
