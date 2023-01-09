from multiprocessing import Process

import cv2
import time

from vidgear.gears import CamGear, StreamGear

from live_streaming_server.models.base import OpenVINOBase

from .config import Config
from .utils import is_darwin, is_windows


class StreamingService(Process):
    def __init__(self, model: OpenVINOBase, config: Config) -> None:
        super().__init__(daemon=True)
        self.config = config
        self._model = model
        self._camera_params = {
            "CAP_PROP_FRAME_WIDTH": 1280,
            "CAP_PROP_FRAME_HEIGHT": 720,
            "CAP_PROP_FPS": 30,
            "CAP_PROP_FOURCC": cv2.VideoWriter_fourcc(*"MJPG"),
        }
        self._stream_params = {
            "-streams": [
                {
                    "-resolution": "640x480",
                    "-framerate": 30.0,
                },  # Stream3: 640x480 at 30fps framerate
                {
                    "-resolution": "480x360",
                    "-framerate": 30.0,
                },  # Stream3: 480x360 at 30fps framerate
            ],
            "-livestream": True,
            "-clear_prev_assets": True,
            "-hls_init_time": 2,
            "-hls_time": 1,
            "-hls_list_size": 10,
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
        self._streamer = StreamGear(
            output=self.config.m3u8_file_path,
            format="hls",
            custom_ffmpeg=self.config.FFMPEG_PATH,
            logging=self.config.STREAMING_DEBUG,
            **self._stream_params
        )

        try:
            start_time = time.time()
            counter = 0
            while True:
                # read frames from stream
                frame = self._stream.read()
                if frame is None:
                    break

                # in-place inference
                self._model.infer_image(frame)

                self._streamer.stream(frame)

                if not self.config.STREAMING_DEBUG:
                    counter += 1
                    if (time.time() - start_time) > 1:
                        print("FPS: ", counter / (time.time() - start_time))
                        counter = 0
                        start_time = time.time()

                if self.config.SHOW_MODEL_OUTPUT:
                    cv2.imshow("Capture", frame)
                    cv2.waitKey(1)
        finally:
            self._stop()

        print("Finished!")

    def _stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()

        if self._streamer is not None:
            self._streamer.terminate()
