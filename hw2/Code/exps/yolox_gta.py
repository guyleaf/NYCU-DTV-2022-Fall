#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os

import torch
import torch.distributed as dist

from yolox.data import get_yolox_datadir
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        # ---------------- model config ---------------- #
        # detect classes number of model
        self.num_classes = 80
        # factor of model depth
        self.depth = 1.00
        # factor of model width
        self.width = 1.00
        # activation name. For example, if using "relu", then "silu" will be replaced to "relu".
        self.act = "silu"

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        # If your training process cost many memory, reduce this value.
        self.data_num_workers = 4
        self.input_size = (416, 416)  # (height, width)
        # Actual multiscale ranges: [640 - 5 * 32, 640 + 5 * 32].
        # To disable multiscale training, set the value to 0.
        self.multiscale_range = 5
        # You can uncomment this line to specify a multiscale range
        # self.random_size = (14, 26)
        # dir of dataset images, if data_dir is None, this project will use `datasets` dir
        self.data_dir = os.path.join(get_yolox_datadir(), "GTA")
        # name of annotation file for training
        del self.train_ann
        # name of annotation file for evaluation
        del self.val_ann
        # name of annotation file for testing
        del self.test_ann

        # --------------- transform config ----------------- #
        # applying mosaic aug
        self.mosaic_prob = 1.0
        self.mosaic_scale = (0.1, 2)
        # applying mixup aug
        self.mixup_prob = 1.0
        self.enable_mixup = True
        self.mixup_scale = (0.5, 1.5)
        # applying color jitter aug
        self.jitter_prob = 1.0
        self.jitter_brightness = (0.2,)
        self.jitter_contrast = (0.2,)
        self.jitter_saturation = (0.2,)
        self.jitter_hue = (0.2,)
        # prob of applying flip aug
        self.flip_prob = 0.5
        # applying padding if needed
        self.padding_value: int = (0,)
        # minimum (area of scaled bbox / area of original bbox)
        self.bbox_min_visibility: float = 0.2

        # affine transformation
        # rotation angle range, for example, if set to 2, the true range is (-2, 2)
        self.degrees = 10.0
        # translate range, for example, if set to 0.1, the true range is (-0.1, 0.1)
        self.translate = 0.1
        # shear angle range, for example, if set to 2, the true range is (-2, 2)
        self.shear = 2.0

        # --------------  training config --------------------- #
        # epoch number used for warmup
        self.warmup_epochs = 5
        # max training epoch
        self.max_epoch = 300
        # minimum learning rate during warmup
        self.warmup_lr = 0
        self.min_lr_ratio = 0.05
        # learning rate for one image. During training, lr will multiply batchsize.
        self.basic_lr_per_img = 0.01 / 64.0
        # name of LRScheduler
        self.scheduler = "yoloxwarmcos"
        # last #epoch to close augmention like mosaic
        self.no_aug_epochs = 15
        # apply EMA during training
        self.ema = True

        # weight decay of optimizer
        self.weight_decay = 5e-4
        # momentum of optimizer
        self.momentum = 0.9
        # log period in iter, for example,
        # if set to 1, user could see log every iteration.
        self.print_interval = 10
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

        # -----------------  testing config ------------------ #
        # output image size during evaluation/test
        self.test_size = (416, 416)
        # confidence threshold during evaluation/test,
        # boxes whose scores are less than test_conf will be filtered
        self.test_conf = 0.01
        # nms threshold
        self.nmsthre = 0.65

    def get_data_loader(
        self,
        batch_size: int,
        is_distributed: bool,
        no_aug: bool = False,
        _: bool = False,
    ):
        from yolox.data import (
            GTAVideoDataset,
            GTATrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDataset,
            worker_init_reset_seed,
        )
        from yolox.utils import (
            wait_for_the_master,
            get_local_rank,
        )

        local_rank = get_local_rank()

        with wait_for_the_master(local_rank):
            dataset = GTAVideoDataset(
                data_dir=self.data_dir,
                image_set="train",
                img_size=self.input_size,
                preproc=GTATrainTransform(
                    img_size=self.input_size,
                    jitter_prob=self.jitter_prob,
                    flip_prob=self.flip_prob,
                    jitter_brightness=self.jitter_brightness,
                    jitter_contrast=self.jitter_contrast,
                    jitter_saturation=self.jitter_saturation,
                    jitter_hue=self.jitter_hue,
                    padding_value=self.padding_value,
                    bbox_min_visibility=self.bbox_min_visibility,
                ),
            )

        self.dataset = MosaicDataset(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=GTATrainTransform(
                img_size=self.input_size,
                jitter_prob=self.jitter_prob,
                flip_prob=self.flip_prob,
                jitter_brightness=self.jitter_brightness,
                jitter_contrast=self.jitter_contrast,
                jitter_saturation=self.jitter_saturation,
                jitter_hue=self.jitter_hue,
                padding_value=self.padding_value,
                bbox_min_visibility=self.bbox_min_visibility,
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
            "shuffle": True,
        }
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(
        self, batch_size, is_distributed, testdev=False, legacy=False
    ):
        from yolox.data import GTAVideoDataset, GTAValTransform

        valdataset = VOCDetectionDataset(
            data_dir=os.path.join(get_yolox_datadir(), "VOCdevkit"),
            image_sets=[("2007", "test")],
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(
            valdataset, **dataloader_kwargs
        )

        return val_loader

    def get_evaluator(
        self, batch_size, is_distributed, testdev=False, legacy=False
    ):
        from yolox.evaluators import VOCEvaluator

        val_loader = self.get_eval_loader(
            batch_size, is_distributed, testdev, legacy
        )
        evaluator = VOCEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )
        return evaluator
