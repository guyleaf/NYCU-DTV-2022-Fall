from concurrent.futures import Future
from multiprocessing.connection import Connection
import time
import copy
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from multiprocessing import Process
from pathlib import Path
from threading import Thread
from typing import Any

import cv2
import numpy as np
from vidgear.gears import CamGear, StreamGear

from live_streaming_server.models.base import OpenVINOBase

from .config import Config
from .utils import is_darwin, is_windows


class StreamingService(Process):
    def __init__(
        self,
        adding_streamer_pipe: Connection,
        updating_classes_pipe: Connection,
        model: OpenVINOBase,
        config: Config,
    ) -> None:
        super().__init__(daemon=True)
        self.config = config
        self._model = model
        self._streamings = {}
        self._adding_streamer_pipe = adding_streamer_pipe
        self._updating_classes_pipe = updating_classes_pipe
        self._camera_params = {
            "CAP_PROP_FRAME_WIDTH": 1280,
            "CAP_PROP_FRAME_HEIGHT": 720,
            "CAP_PROP_FPS": 30,
            "CAP_PROP_FOURCC": cv2.VideoWriter_fourcc(*"MJPG"),
        }
        self._stream_params = {
            "-streams": [
                # {
                #     "-resolution": "960x540",
                #     "-framerate": 13.0,
                # },  # Stream3: 960x540 at 24fps framerate
                {
                    "-resolution": "640x360",
                    "-framerate": 30.0,
                },  # Stream3: 640x360 at 24fps framerate
            ],
            "-livestream": True,
            "-clear_prev_assets": True,
            "-hls_init_time": 4,
            "-hls_time": 1,
            "-hls_list_size": 15,
            "-vcodec": "libx264",
        }

        if is_darwin():
            self._stream_params.update(
                {
                    "-vcodec": "h264_videotoolbox",
                    "-realtime": "true",
                    "-allow_sw": "true",
                    "-profile:v": 3,
                    "-level": 52,
                    "-coder": 2,
                }
            )

    def create_streamer(self, id: str):
        output = self.config.get_m3u8_file_path(id)
        folder = Path(output).parent.absolute()
        folder.mkdir(exist_ok=True)

        self._streamings[id] = {
            "streamer": StreamGear(
                output=output,
                format="hls",
                custom_ffmpeg=self.config.FFMPEG_PATH,
                logging=self.config.STREAMING_DEBUG,
                **copy.deepcopy(self._stream_params)
            ),
            "classes": {
                str(id): True for id in range(len(self._model.classes))
            },
        }
        print("Streaming is created.")

    def update_classes(self, id: str, new_classes: dict[str, bool]):
        self._streamings[id]["classes"] = new_classes

    def _filter_bboxes(self, box_info: np.ndarray, classes: dict[str, bool]):
        for *xyxy, score, class_id in box_info:
            class_id = int(class_id)
            if classes[str(class_id)]:
                yield (xyxy, score, class_id)

    def send_to_streaming(
        self, streaming: dict[str, Any], frame: cv2.Mat, box_info: np.ndarray
    ):
        frame = frame.copy()
        self._model.draw(
            frame,
            self._filter_bboxes(box_info, streaming["classes"])
        )
        streamer: StreamGear = streaming["streamer"]
        streamer.stream(frame)

    def test(self, result: Future):
        message = result.result()
        if self.config.STREAMING_DEBUG:
            print(message)

    def run(self) -> None:
        self._model.initialize()

        backend = cv2.CAP_V4L2
        if is_windows():
            backend = cv2.CAP_DSHOW
        elif is_darwin():
            backend = cv2.CAP_AVFOUNDATION

        self._stream = CamGear(
            source=self.config.DEVICE,
            backend=backend,
            logging=self.config.STREAMING_DEBUG,
            **self._camera_params
        ).start()

        self._stream_params["-input_framerate"] = self._stream.framerate

        try:
            start_time = time.time()
            counter = 0

            with ThreadPoolExecutor(max_workers=4) as executor:
                while True:
                    # read frames from stream
                    frame = self._stream.read()
                    if frame is None:
                        break

                    box_info = self._model.infer_image(frame)

                    for streaming in self._streamings.values():
                        # self.send_to_streaming(streaming, frame, box_info)
                        result = executor.submit(
                            self.send_to_streaming, streaming, frame, box_info
                        )
                        result.add_done_callback(self.test)

                    if not self.config.STREAMING_DEBUG:
                        counter += 1
                        if (time.time() - start_time) > 1:
                            print(
                                "FPS: ", counter / (time.time() - start_time)
                            )
                            counter = 0
                            start_time = time.time()

                    if self.config.SHOW_MODEL_OUTPUT:
                        cv2.imshow("Capture", frame)
                        cv2.waitKey(1)

                    if self._adding_streamer_pipe.poll(timeout=0.001):
                        data = self._adding_streamer_pipe.recv()
                        Thread(target=self.create_streamer, args=data).start()

                    if self._updating_classes_pipe.poll(timeout=0.001):
                        data = self._updating_classes_pipe.recv()
                        Thread(target=self.update_classes, args=data).start()

        finally:
            self._stop()

        print("Finished!")

    def _stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()

        for streaming in self._streamings.values():
            streamer: StreamGear = streaming["streamer"]
            streamer.terminate()
