# Reference: https://github.com/OpenVINO-dev-contest/YOLOv7_OpenVINO_cpp-python/blob/360ac86f6d82aed187b5e2ed57a70d22ad59f64f/python/yolov7.py # noqa: E501

import cv2
import numpy as np

from .classes import COCO2017_CLASSES
from .base import OpenVINOBase


class Yolov7OpenVINO(OpenVINOBase):
    def __init__(
        self,
        model: str,
        device: str,
        pre_api: bool,
        grid: bool,
        end2end: bool,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
    ):
        super().__init__(
            model,
            device,
            pre_api,
            COCO2017_CLASSES,
            (640, 640),
        )
        # set the hyperparameters
        self.grid = grid
        self.end2end = end2end
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.class_num = len(self._classes)
        self.stride = [8, 16, 32]
        self.anchor_list = [
            [12, 16, 19, 36, 40, 28],
            [36, 75, 76, 55, 72, 146],
            [142, 110, 192, 243, 459, 401],
        ]
        self.anchor = (
            np.array(self.anchor_list).astype(float).reshape(3, -1, 2)
        )
        area = self.img_size[0] * self.img_size[1]
        self.size = [
            int(area / self.stride[0] ** 2),
            int(area / self.stride[1] ** 2),
            int(area / self.stride[2] ** 2),
        ]
        self.feature = [
            [int(j / self.stride[i]) for j in self.img_size] for i in range(3)
        ]

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y

        return y

    def nms(self, prediction, conf_thres, iou_thres):
        predictions = np.squeeze(prediction[0])

        # Filter out object confidence scores below threshold
        obj_conf = predictions[:, 4]
        predictions = predictions[obj_conf > conf_thres]
        obj_conf = obj_conf[obj_conf > conf_thres]

        # Multiply class confidence with bounding box confidence
        predictions[:, 5:] *= obj_conf[:, np.newaxis]

        # Get the scores
        scores = np.max(predictions[:, 5:], axis=1)

        # Filter out the objects with a low score
        valid_scores = scores > conf_thres
        predictions = predictions[valid_scores]
        scores = scores[valid_scores]

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 5:], axis=1)

        # Get bounding boxes for each object
        boxes = self.xywh2xyxy(predictions[:, :4])

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        # indices = nms(boxes, scores, self.iou_threshold)
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(), conf_thres, iou_thres
        )

        return boxes[indices], scores[indices], class_ids[indices]

    def clip_coords(self, boxes, img_shape):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        boxes[:, 0].clip(0, img_shape[1])  # x1
        boxes[:, 1].clip(0, img_shape[0])  # y1
        boxes[:, 2].clip(0, img_shape[1])  # x2
        boxes[:, 3].clip(0, img_shape[0])  # y2

    def scale_coords(self, img1_shape, img0_shape, coords, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        # gain  = old / new
        if ratio_pad is None:
            gain = min(
                img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
            )
            padding = (img1_shape[1] - img0_shape[1] * gain) / 2, (
                img1_shape[0] - img0_shape[0] * gain
            ) / 2
        else:
            gain = ratio_pad[0][0]
            padding = ratio_pad[1]
        coords[:, [0, 2]] -= padding[0]  # x padding
        coords[:, [1, 3]] -= padding[1]  # y padding
        coords[:, :4] /= gain
        self.clip_coords(coords, img0_shape)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def postprocess(self, infer_request, src_size):
        if self.end2end:
            results = np.expand_dims(
                infer_request.get_output_tensor(0).data, axis=0
            )
        elif self.grid:
            results = np.expand_dims(
                infer_request.get_output_tensor(0).data[0], axis=0
            )
        else:
            output = []
            # Get the each feature map's output data
            output.append(
                self.sigmoid(
                    infer_request.get_output_tensor(0)
                    .data[0]
                    .reshape(-1, self.size[0] * 3, 5 + self.class_num)
                )
            )
            output.append(
                self.sigmoid(
                    infer_request.get_output_tensor(1)
                    .data[0]
                    .reshape(-1, self.size[1] * 3, 5 + self.class_num)
                )
            )
            output.append(
                self.sigmoid(
                    infer_request.get_output_tensor(2)
                    .data[0]
                    .reshape(-1, self.size[2] * 3, 5 + self.class_num)
                )
            )

            # Postprocessing
            grid = []
            for _, f in enumerate(self.feature):
                grid.append([[i, j] for j in range(f[0]) for i in range(f[1])])

            result = []
            for i in range(3):
                src = output[i]
                xy = src[..., 0:2] * 2.0 - 0.5
                wh = (src[..., 2:4] * 2) ** 2
                dst_xy = []
                dst_wh = []
                for j in range(3):
                    left = j * self.size[i]
                    right = (j + 1) * self.size[i]
                    dst_xy.append(
                        (xy[:, left:right, :] + grid[i]) * self.stride[i]
                    )
                    dst_wh.append(wh[:, left:right, :] * self.anchor[i][j])
                src[..., 0:2] = np.concatenate(
                    (dst_xy[0], dst_xy[1], dst_xy[2]), axis=1
                )
                src[..., 2:4] = np.concatenate(
                    (dst_wh[0], dst_wh[1], dst_wh[2]), axis=1
                )
                result.append(src)
            results = np.concatenate(result, 1)

        if self.end2end:
            results = results[0]
            _, boxes, class_ids, scores = np.split(results, [1, 5, 6], axis=-1)
        else:
            boxes, scores, class_ids = self.nms(
                results, self.conf_thres, self.iou_thres
            )

        img_shape = self.img_size
        self.scale_coords(img_shape, src_size, boxes)

        scores = np.expand_dims(scores, axis=-1)
        class_ids = np.expand_dims(class_ids, axis=-1)
        return np.concatenate((boxes, scores, class_ids), axis=-1)
