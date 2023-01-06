from typing import Literal

from .base import OPENVINO_BASE
from .yolov7 import YOLOV7_OPENVINO

YOLOX7_MODELS = Literal[
    "yolov7/yolov7-tiny",
    "yolov7/yolov7-tiny_int8",
    "yolov7/yolov7-tiny-end2end",
    "yolov7/yolov7-tiny-end2end_int8",
]

MODELS = Literal[YOLOX7_MODELS]

__all__ = ["OPENVINO_BASE", "YOLOV7_OPENVINO", "MODELS"]
