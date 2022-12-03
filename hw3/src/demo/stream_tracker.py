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

from .captures import VideoCapture


class StreamTrackerManager:
    def __init__(
        self,
        tracker: Tracker,
        transform: Compose,
        interpolate: bool,
        _log,
    ) -> None:
        self._log = _log
        self._interpolate = interpolate
        self._transform = transform
        self._tracker = tracker
        self._stream_tracker = None

    def __del__(self):
        self.stop_tracker()

    def stop_tracker(self):
        if self._stream_tracker is not None:
            self._log.info("Stopping the tracker...")
            self._stream_tracker.stop()
            self._stream_tracker.join()
            del self._stream_tracker
            self._stream_tracker = None

    def start_tracker(self, data_source: Union[int, str], output_queue: Queue):
        self._log.info("Starting a tracker...")
        self._stream_tracker = StreamTracker(
            self._tracker,
            data_source,
            self._transform,
            output_queue,
            self._interpolate,
            self._log,
        )
        self._stream_tracker.setDaemon(True)
        self._stream_tracker.start()


class StreamTracker(threading.Thread):
    def __init__(
        self,
        tracker: Tracker,
        cap: VideoCapture,
        data_transform: Compose,
        output_queue: Queue,
        interpolate: bool,
        log,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._output_queue = output_queue
        self._interpolate = interpolate
        self._log = log
        self._cap = cap
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
        )

    def stop(self):
        self._stop_event.set()

    def run(self) -> None:
        self._tracker.reset()

        # FIXME: bad pattern to stop the thread, need to be rewritten
        while not self._stop_event.is_set():
            image = self._cap.read()
            if image is None:
                break

            # self._log.info("Tracking...")
            frame_data = self._preprocess(image)
            with torch.no_grad():
                tracks = self._tracker.step(frame_data)

            self._output_queue.put(
                dict(image=image, tracks=tracks), timeout=30
            )


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

        self._input_queue = Queue(maxsize=input_queue_size)
        self._output_queue = Queue(maxsize=output_queue_size)
        self._input_filters = Queue()
        self._filters = []
        self._log.info("Initilized StreamOutputPipeline.")
        # self._buffers = []

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

        image = plot_single_frame(
            tracks,
            image,
            self._write_images,
            self._cmap,
            self._generate_attention_maps,
        )
        return image

    def run(self) -> None:
        while True:
            if not self._input_filters.empty():
                self._filters = self._input_filters.get_nowait()
                self._input_filters.task_done()

            if self._input_queue.empty():
                continue

            result = self._input_queue.get_nowait()
            self._input_queue.task_done()

            frame = self._post_process(result)
            # self._buffers.append(frame)

            # if full, skip it
            if self._output_queue.full():
                time.sleep(0.1)
            else:
                self._output_queue.put_nowait(frame)
