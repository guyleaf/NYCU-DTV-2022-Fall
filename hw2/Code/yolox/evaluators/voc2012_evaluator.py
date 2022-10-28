from loguru import logger
from tqdm import tqdm

import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
)

from collections import ChainMap
import sys
import time

from .pascalvoc_2012.bounding_box import BoundingBox
from .pascalvoc_2012.bounding_boxes import BoundingBoxes
from .pascalvoc_2012.evaluator import Evaluator as PascalVOC2012Evaluator
from .pascalvoc_2012.utils import BBFormat, BBType


class VOC2012Evaluator:
    """
    VOC 2012 Evaluation class.
    Use PASCAL VOC 2012 metrics implemented by https://github.com/rafaelpadilla/Object-Detection-Metrics
    """

    def __init__(
        self,
        dataloader: DataLoader,
        img_size: int,
        conf_threshold: float,
        nms_threshold: float,
        num_classes: int,
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            conf_threshold (float): confidence threshold ranging from 0 to 1,
                                    which is defined in the config file.
        """
        self._dataloader = dataloader
        self._img_size = img_size
        self._conf_threshold = conf_threshold
        self._nms_threshold = nms_threshold
        self._iou_thresholds = list(range(0.5, 1.0, 0.05))
        self._num_classes = num_classes
        self._num_images = len(dataloader.dataset)
        self._evaluator = PascalVOC2012Evaluator()

    def _wrap_gt_bboxes(self) -> BoundingBoxes:
        wrapped_bboxes = BoundingBoxes()
        for i in range(self._num_images):
            _, target, _, img_id = self._dataloader.dataset.pull_item(i)
            for x_min, y_min, x_max, y_max, class_id in target:
                bbox = BoundingBox(
                    img_id,
                    class_id,
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                    bbType=BBType.GroundTruth,
                    format=BBFormat.XYX2Y2,
                )
                wrapped_bboxes.addBoundingBox(bbox)
        return wrapped_bboxes

    def _wrap_pred_bboxes(
        self, all_boxes: list[dict[str, np.ndarray]]
    ) -> BoundingBoxes:
        wrapped_bboxes = BoundingBoxes()
        for class_id, imgs in enumerate(all_boxes):
            for img_id, bboxes in imgs.items():
                for (score, x_min, y_min, x_max, y_max) in bboxes:
                    bbox = BoundingBox(
                        img_id,
                        class_id,
                        x_min,
                        y_min,
                        x_max,
                        y_max,
                        bbType=BBType.Detected,
                        classConfidence=score,
                        format=BBFormat.XYX2Y2,
                    )
                    wrapped_bboxes.addBoundingBox(bbox)
        return wrapped_bboxes

    def _convert_preds_to_voc_format(
        self, outputs: torch.Tensor, img_infos, ids: list[str]
    ):
        predictions = {}
        for (output, img_h, img_w, img_id) in zip(
            outputs, img_infos[0], img_infos[1], ids
        ):
            if output is None:
                predictions[img_id] = (None, None, None)
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self._img_size[0] / float(img_h),
                self._img_size[1] / float(img_w),
            )
            bboxes /= scale

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]

            predictions[img_id] = (bboxes, cls, scores)
        return predictions

    def _collect_metrics(self, bboxes: BoundingBoxes):
        mAPs = []
        for iou in self._iou_thresholds:
            metrics = self._evaluator.GetPascalVOCMetrics(bboxes, iou)
            mAP = np.mean([c["AP"] for c in metrics])
            mAPs.append(mAP)
        summary = ",\n".join(
            [
                "mAP%d: %.2f" % (self._iou_thresholds[i] * 100, mAP)
                for i, mAP in enumerate(mAPs)
            ]
        )
        return mAPs, summary

    def _evaluate_prediction(self, data_dict: dict, statistics: torch.Tensor):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = (
            1000 * inference_time / (n_samples * self._dataloader.batch_size)
        )
        a_nms_time = (
            1000 * nms_time / (n_samples * self._dataloader.batch_size)
        )

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "NMS", "inference"],
                    [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
                )
            ]
        )

        info = time_info + "\n"

        all_boxes = [{} for _ in range(self._num_classes)]
        for i, (img_id, (bboxes, cls, scores)) in enumerate(data_dict.items()):
            if bboxes is None:
                for j in range(self._num_classes):
                    all_boxes[j][img_id] = np.empty([0, 5], dtype=np.float32)
                continue
            for j in range(self._num_classes):
                mask_c = cls == j
                if sum(mask_c) == 0:
                    all_boxes[j][img_id] = np.empty([0, 5], dtype=np.float32)
                    continue

                c_dets = torch.cat((scores.unsqueeze(1), bboxes), dim=1)
                all_boxes[j][img_id] = c_dets[mask_c].numpy()

            sys.stdout.write(
                "im_eval: {:d}/{:d} \r".format(i + 1, self._num_images)
            )
            sys.stdout.flush()

        gt_bboxes = self._wrap_gt_bboxes()
        bboxes = self._wrap_bboxes(all_boxes)
        bboxes.addBoundingBoxes(gt_bboxes)
        mAPs, summary = self._collect_metrics(bboxes)
        # metrics = {
        #     "PASCALAP50": mAPs[0],
        #     "PASCALAP50_95": np.mean(mAPs),
        #     "LOSS": "",
        # }
        return mAPs[0], np.mean(mAPs), info + summary + "\n"

    @torch.no_grad()
    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
    ):
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = {}
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self._dataloader) - 1, 1)

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        for cur_iter, (imgs, _, img_infos, ids) in enumerate(
            progress_bar(self._dataloader)
        ):
            imgs = imgs.type(tensor_type)

            # skip the last iters since batchsize might be not enough for batch inference
            is_time_record = cur_iter < len(self._dataloader) - 1
            if is_time_record:
                start = time.time()

            outputs = model(imgs)
            if decoder is not None:
                outputs = decoder(outputs, dtype=outputs.type())

            if is_time_record:
                infer_end = time_synchronized()
                inference_time += infer_end - start

            # [x1, y1, x2, y2, obj_conf, class_conf, class_idx]
            outputs = postprocess(
                outputs,
                self._num_classes,
                self._conf_threshold,
                self._nms_threshold,
            )
            if is_time_record:
                nms_end = time_synchronized()
                nms_time += nms_end - infer_end

            data_list.update(
                self._convert_preds_to_voc_format(outputs, img_infos, ids)
            )

        statistics = torch.tensor(
            [inference_time, nms_time, n_samples], dtype=torch.float32
        ).cuda()
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = ChainMap(*data_list)
            dist.reduce(statistics, dst=0)

        eval_results = self._evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results
