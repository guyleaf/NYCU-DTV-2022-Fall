import random
import cv2
import numpy as np
from openvino.preprocess import ColorFormat, PrePostProcessor
from openvino.runtime import Core, Layout, PartialShape

from .utils import get_model_path


class OpenVINOBase:
    def __init__(
        self,
        model: str,
        device: str,
        pre_api: bool,
        classes: list[str],
        img_size: tuple[int],
    ) -> None:
        # set the hyperparameters
        self._classes = classes
        self.img_size = img_size
        self.pre_api = pre_api
        self.device = device

        self._model = model
        self._colors = [
            [random.randint(0, 255) for _ in range(3)] for _ in self._classes
        ]

    @property
    def classes(self):
        return self._classes.copy()

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (
            new_shape[1] - new_unpad[0],
            new_shape[0] - new_unpad[1],
        )  # wh padding

        # divide padding into 2 sides
        dw /= 2
        dh /= 2

        # resize
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        # add border
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )

        return img

    def initialize(self):
        ie = Core()
        model_path = get_model_path(self._model)
        self.model = ie.read_model(model_path)
        self.input_layer = self.model.input(0)
        new_shape = PartialShape([1, 3, self.img_size[0], self.img_size[1]])
        self.model.reshape({self.input_layer.any_name: new_shape})
        if self.pre_api:
            # Preprocessing API
            ppp = PrePostProcessor(self.model)
            # Declare section of desired application's input format
            ppp.input().tensor().set_layout(Layout("NHWC")).set_color_format(
                ColorFormat.BGR
            )
            # Here, it is assumed that the model has "NCHW" layout for input.
            ppp.input().model().set_layout(Layout("NCHW"))
            # Convert current color format (BGR) to RGB
            ppp.input().preprocess().convert_color(ColorFormat.RGB).scale(
                [255.0, 255.0, 255.0]
            )
            self.model = ppp.build()
            print(f"Dump preprocessor: {ppp}")

        print(f"Inference device: {self.device}")
        config = {"PERFORMANCE_HINT": "THROUGHPUT"}
        self.compiled_model = ie.compile_model(
            model=self.model, device_name=self.device, config=config
        )

    def plot_one_box(
        self, x, img, color=None, label=None, line_thickness=None
    ):
        # Plots one bounding box on image img
        tl = (
            line_thickness
            or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
        )  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[
                0
            ]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(
                img,
                label,
                (c1[0], c1[1] - 2),
                0,
                tl / 3,
                [225, 255, 255],
                thickness=tf,
                lineType=cv2.LINE_AA,
            )

    def draw(self, img, box_info):
        for xyxy, conf, cls in box_info:
            cls = int(cls)
            conf = float(conf)
            label = f"{self._classes[cls]} {conf:.1f}"
            self.plot_one_box(
                xyxy,
                img,
                label=label,
                color=self._colors[cls],
                line_thickness=3,
            )

    def postprocess(self, infer_request, src_size):
        raise NotImplementedError("The method postprocess is not implemented.")

    # return: [x, y, x, y, score, class_ind]
    def infer_image(self, src_img: cv2.Mat) -> np.ndarray:
        img = self.letterbox(src_img, self.img_size)

        src_size = src_img.shape[:2]
        img = img.astype(dtype=np.float32)
        if not self.pre_api:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB
            img /= 255.0
            img = img.transpose(2, 0, 1)  # NHWC to NCHW
            img = np.ascontiguousarray(img)
        input_image = np.expand_dims(img, 0)

        infer_request = self.compiled_model.create_infer_request()
        infer_request.start_async({self.input_layer.any_name: input_image})
        infer_request.wait()
        return self.postprocess(infer_request, src_size)
