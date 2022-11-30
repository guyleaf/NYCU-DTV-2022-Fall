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

import time
from collections import ChainMap

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
        img_size: tuple[int, int],
        conf_threshold: float,
        nms_threshold: float,
        num_classes: int,
        train_dataloader: DataLoader = None,
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (tuple[int, int]): image size after preprocess.
            conf_threshold (float): confidence threshold ranging from 0 to 1,
                                    which is defined in the config file.
        """
        self._dataloader = dataloader
        self._train_dataloader = train_dataloader
        self._img_size = img_size
        self._conf_threshold = conf_threshold
        self._nms_threshold = nms_threshold
        self._iou_thresholds = np.linspace(
            0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
        )
        self._num_classes = num_classes
        self._evaluator = PascalVOC2012Evaluator()

    def _wrap_gt_bboxes(
        self, dataloader: DataLoader
    ) -> tuple[BoundingBoxes, dict[int, np.ndarray]]:
        bboxes = {}
        wrapped_bboxes = BoundingBoxes()
        for i in range(len(dataloader.dataset)):
            _, target, img_info, img_id = dataloader.dataset.pull_item(i)
            if isinstance(img_id, (np.ndarray, torch.Tensor)):
                img_id = int(img_id)

            scale = min(
                self._img_size[0] / img_info[0],
                self._img_size[1] / img_info[1],
            )
            target[:, :4] /= scale

            tmp = []
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
                tmp.append([class_id, x_min, y_min, x_max, y_max])
                wrapped_bboxes.addBoundingBox(bbox)
            if len(tmp) != 0:
                bboxes[img_id] = np.array(tmp, dtype=np.float32)
        return wrapped_bboxes, bboxes

    def _wrap_pred_bboxes(
        self, data_dict: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]
    ) -> BoundingBoxes:
        wrapped_bboxes = BoundingBoxes()
        # data_dict[] = (cls, scores, bboxes)
        for img_id, output in data_dict.items():
            for cls, score, x_min, y_min, x_max, y_max in output:
                bbox = BoundingBox(
                    img_id,
                    int(cls),
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
        self,
        outputs: torch.Tensor,
        img_infos: tuple[torch.Tensor, torch.Tensor],
        ids: list[str],
    ):
        predictions = {}
        for (output, img_h, img_w, img_id) in zip(
            outputs, img_infos[0], img_infos[1], ids
        ):
            img_id = int(img_id)
            if output is None:
                predictions[img_id] = np.empty([0, 6], dtype=np.float32)
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self._img_size[0] / float(img_h),
                self._img_size[1] / float(img_w),
            )
            bboxes /= scale

            cls = output[:, 6].unsqueeze(dim=-1)
            scores = output[:, 4] * output[:, 5]
            scores = scores.unsqueeze(dim=-1)

            predictions[img_id] = torch.cat(
                (cls, scores, bboxes), dim=1
            ).numpy()
        return predictions

    def _collect_metrics(self, bboxes: BoundingBoxes) -> list[float]:
        mAPs = []
        for iou in self._iou_thresholds:
            metrics = self._evaluator.GetPascalVOCMetrics(bboxes, iou)
            mAP = np.mean([c["AP"] for c in metrics])
            mAPs.append(mAP)
        return mAPs

    # def _write_text_files(
    #     self,
    #     data_dict: dict[int, np.ndarray],
    #     is_gt=False,
    # ):
    #     dir = os.path.join(
    #         self._output_dir, "results", "gts" if is_gt else "preds"
    #     )
    #     os.makedirs(dir, exist_ok=True)
    #     dtypes = {
    #         "class": "int32",
    #         "x_min": "int32",
    #         "y_min": "int32",
    #         "x_max": "int32",
    #         "y_max": "int32",
    #     }
    #     columns = ["class", "score", "x_min", "y_min", "x_max", "y_max"]
    #     if is_gt:
    #         columns.pop(1)

    #     for img_id, output in data_dict.items():
    #         filename = os.path.join(dir, f"{img_id}.txt")
    #         data = pd.DataFrame(
    #             output,
    #             columns=columns,
    #         ).astype(dtypes)
    #         data.to_csv(
    #             filename, sep=" ", index=False, header=False, encoding="utf-8"
    #         )

    def _evaluate_prediction(
        self,
        gt_bboxes: tuple[BoundingBoxes, dict[int, np.ndarray]],
        data_dict: dict[int, np.ndarray],
        statistics: torch.Tensor,
    ) -> tuple[list[float], str]:
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

        bboxes = self._wrap_pred_bboxes(data_dict)
        bboxes.addBoundingBoxes(gt_bboxes[0])
        mAPs = self._collect_metrics(bboxes)
        return mAPs, info

    def _evaluate(
        self,
        model,
        dataloader: DataLoader,
        distributed=False,
        half=False,
        decoder=None,
    ) -> tuple[list[float], str]:
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        data_list = {}
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(dataloader) - 1, 1)

        for cur_iter, (imgs, _, img_infos, ids) in enumerate(
            progress_bar(dataloader)
        ):
            imgs = imgs.type(tensor_type)

            # skip the last iters since batchsize might be not enough for batch inference
            is_time_record = cur_iter < len(dataloader) - 1
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

        gt_bboxes = self._wrap_gt_bboxes(dataloader)
        mAPs, summary = self._evaluate_prediction(
            gt_bboxes, data_list, statistics
        )
        synchronize()
        return mAPs, summary

    @torch.no_grad()
    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        eval_train=False,
    ):
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        val_mAPs, summary = self._evaluate(
            model, self._dataloader, distributed, half, decoder
        )

        if eval_train:
            assert self._train_dataloader is not None
            train_mAPs, _ = self._evaluate(
                model, self._train_dataloader, distributed, half, decoder
            )
            generator = zip(train_mAPs, val_mAPs)
        else:
            generator = val_mAPs

        logs = {}
        for i, val_mAP in enumerate(generator):
            iou = int(round(self._iou_thresholds[i], 2) * 100)
            if eval_train:
                train_mAP, val_mAP = val_mAP
                logs[f"train/VOC2012_mAP{iou}"] = train_mAP

            logs[f"val/VOC2012_mAP{iou}"] = val_mAP
            if i != 0:
                summary += ", "
            summary += "mAP%d: %.3f" % (iou, val_mAP)
        summary += "\n"

        if eval_train:
            logs["train/VOC2012_mAP50:95"] = np.mean(train_mAPs)

        mAP50_95 = np.mean(val_mAPs)
        logs["val/VOC2012_mAP50:95"] = mAP50_95

        synchronize()
        return mAP50_95, val_mAPs[0], logs, summary
