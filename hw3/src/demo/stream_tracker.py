# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import threading
import time
from queue import Queue
from typing import Any, Dict, Union

import numpy as np
import torch
from PIL import Image

from trackformer.datasets.transforms import Compose
from trackformer.models.tracker import Tracker
from trackformer.util.track_utils import plot_single_frame, rand_cmap

from .captures import BaseCapture


class StreamTrackerManager:
    def __init__(
        self,
        tracker: Tracker,
        transform: Compose,
        _log,
    ) -> None:
        self._log = _log
        self._transform = transform
        self._tracker = tracker
        self._stream_tracker = None

    def stop_tracker(self):
        if self._stream_tracker is not None:
            self._log.info("Stopping the tracker...")
            self._stream_tracker.stop()
            self._stream_tracker.join()
            del self._stream_tracker
            self._stream_tracker = None

    def start_tracker(self, capture: BaseCapture, output_queue: Queue):
        self._log.info("Starting a tracker...")
        self._stream_tracker = StreamTracker(
            self._tracker,
            capture,
            self._transform,
            output_queue,
            self._log,
        )
        self._stream_tracker.daemon = True
        self._stream_tracker.start()


class StreamTracker(threading.Thread):
    def __init__(
        self,
        tracker: Tracker,
        capture: BaseCapture,
        data_transform: Compose,
        output_queue: Queue,
        log,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._output_queue = output_queue
        self._log = log
        self._capture = capture
        self._data_transform = data_transform
        self._tracker = tracker

        self._stop_event = threading.Event()
        self._log.info("Initilized StreamTracker.")

    def _preprocess(self, image: np.ndarray) -> Dict[str, Any]:
        height_orig, width_orig = image.shape[:2]
        image = Image.fromarray(image)
        img, _ = self._data_transform(image)
        width, height = img.size(2), img.size(1)

        return dict(
            img=torch.unsqueeze(img, 0),
            img_path="",
            dets=torch.tensor([[]]),
            orig_size=torch.as_tensor([[height_orig, width_orig]]),
            size=torch.as_tensor([[int(height), int(width)]]),
        ), (height_orig, width_orig)

    def stop(self):
        self._stop_event.set()

    @torch.no_grad()
    def run(self) -> None:
        self._tracker.reset()
        self._capture.start()

        while not self._stop_event.is_set() and self._capture.running():
            image = self._capture.read()

            # self._log.info("Tracking...")
            frame_data, orig_size = self._preprocess(image)
            tracks = self._tracker.step(frame_data, clear_prev_results=True)

            self._output_queue.put(
                dict(image=image, tracks=tracks, orig_size=orig_size),
                timeout=30,
            )

        self._capture.stop()


class StreamOutputPipeline(threading.Thread):
    def __init__(
        self,
        write_images: Union[bool, str],
        generate_attention_maps: bool,
        max_labels: int,
        log,
        input_queue_size: int = 60,
        output_queue_size: int = 30,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._log = log
        self._write_images = write_images
        self._generate_attention_maps = generate_attention_maps
        self._cmap = rand_cmap(
            max_labels,
            type="bright",
            first_color_black=False,
            last_color_black=False,
        )

        self._stop_event = threading.Event()
        self._input_queue = Queue(maxsize=input_queue_size)
        self._output_queue = Queue(maxsize=output_queue_size)
        self._input_filters = Queue()
        self._filters = []
        self._log.info("Initilized StreamOutputPipeline.")

    @property
    def input_filters(self) -> Queue:
        return self._input_filters

    @property
    def input_queue(self) -> Queue:
        return self._input_queue

    @property
    def output_queue(self) -> Queue:
        return self._output_queue

    def _post_process(self, result: dict):
        image = result["image"]
        tracks = result["tracks"].copy()

        for track_id in self._filters:
            if track_id in tracks:
                del tracks[track_id]

        result["image"] = plot_single_frame(
            tracks,
            image,
            self._write_images,
            self._cmap,
            self._generate_attention_maps,
        )
        return result

    def stop(self):
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.is_set():
            if not self._input_filters.empty():
                self._filters = self._input_filters.get_nowait()
                self._log.info(self._filters)
                self._input_filters.task_done()

            if self._input_queue.empty():
                time.sleep(0.1)
                continue

            result = self._input_queue.get_nowait()
            self._input_queue.task_done()

            result = self._post_process(result)

            self._output_queue.put(result, timeout=30)
