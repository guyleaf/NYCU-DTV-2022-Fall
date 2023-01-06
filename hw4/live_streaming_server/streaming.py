import logging
import platform
from multiprocessing import Process

import cv2
from vidgear.gears import CamGear, StreamGear

from live_streaming_server.models.base import OPENVINO_BASE

from .config import Config


class StreamingService(Process):
    def __init__(self, model: OPENVINO_BASE, config: Config) -> None:
        super().__init__(daemon=True)
        self.config = config
        self._model = model
        self._camera_params = {
            "CAP_PROP_FRAME_WIDTH": 1920,
            "CAP_PROP_FRAME_HEIGHT": 1080,
            "CAP_PROP_FPS": 30,
            "CAP_PROP_FOURCC": cv2.VideoWriter_fourcc(*"MJPG"),
        }
        self._stream_params = {
            "-streams": [
                {
                    "-resolution": "1280x720",
                    "-framerate": 30.0,
                },  # Stream2: 1280x720 at 30fps framerate
                {
                    "-resolution": "640x360",
                    "-framerate": 30.0,
                },  # Stream3: 640x360 at 30fps framerate
                {
                    "-resolution": "320x180",
                    "-framerate": 30.0,
                },  # Stream3: 320x240 at 30fps framerate
            ],
            "-livestream": True,
            "-clear_prev_assets": True,
            "-window_size": 1,
            "-hls_init_time": 1,
            "-hls_time": 1,
            "-hls_list_size": 3,
            "-vcodec": "libx264",
        }

    def run(self) -> None:
        self._model.initialize()

        backend = cv2.CAP_DSHOW
        if platform.system() == "Linux":
            backend = cv2.CAP_V4L2

        self._stream = CamGear(
            source=self.config.DEVICE,
            backend=backend,
            logging=self.config.STREAMING_DEBUG,
            **self._camera_params
        ).start()

        self._stream_params["-input_framerate"] = self._stream.framerate
        self._streamer = StreamGear(
            output=self.config.m3u8_file_path,
            format="hls",
            custom_ffmpeg=self.config.FFMPEG_PATH,
            logging=self.config.STREAMING_DEBUG,
            **self._stream_params
        )

        try:
            while True:
                # read frames from stream
                frame = self._stream.read()
                if frame is None:
                    break

                # in-place inference
                self._model.infer_image(frame)

                self._streamer.stream(frame)
                if self.config.SHOW_MODEL_OUTPUT:
                    cv2.imshow("Capture", frame)
                    cv2.waitKey(1)
        finally:
            self._stop()

        logging.log(logging.INFO, "Finished!")

    def _stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()

        if self._streamer is not None:
            self._streamer.terminate()
