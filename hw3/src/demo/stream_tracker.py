# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import threading
import time
from queue import Queue
from typing import Any, Dict, Union

import numpy as np
import torch

from trackformer.datasets.transforms import Compose
from trackformer.models.tracker import Tracker
from trackformer.util.track_utils import interpolate_tracks

from .captures import VideoCapture


class StreamTracker(threading.Thread):
    def __init__(
        self,
        tracker: Tracker,
        data_source: Union[int, str],
        data_transform: Compose,
        result_queue: Queue,
        interpolate: bool,
        log,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._result_queue = result_queue
        self._interpolate = interpolate
        self._log = log
        self._data_source = data_source
        self._data_transform = data_transform
        self._tracker = tracker

        self._stop_event = threading.Event()

    def _preprocess(self, image: np.ndarray) -> Dict[str, Any]:
        height_orig, width_orig = image.shape[:2]
        img, _ = self._data_transform(image)
        width, height = img.size(2), img.size(1)

        return dict(
            img=img,
            img_path="",
            dets=torch.tensor([]),
            orig_size=torch.as_tensor([height_orig, width_orig]),
            size=torch.as_tensor([int(height), int(width)]),
        )

    def stop(self):
        self._stop_event.set()

    def run(self) -> None:
        cap = VideoCapture(self._data_source)
        self._tracker.reset()

        while not self._stop_event.is_set():
            image = cap.get_frame()
            if image is None:
                break

            start = time.time()

            with torch.no_grad():
                frame_data = self._preprocess(image)
                self._tracker.step(frame_data)

            results = self._tracker.get_results()

            time_total = time.time() - start

            self._log.info(f"NUM ReIDs: {self._tracker.num_reids}")
            self._log.info(f"RUNTIME: {time.time() - start :.2f} s")

            if self._interpolate:
                results = interpolate_tracks(results)

            self._log.info(f"RUNTIME SINGLE FRAME: {time_total:.2f} s")
            self._result_queue.put_nowait(results)
