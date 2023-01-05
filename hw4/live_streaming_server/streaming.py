import logging
import signal
from multiprocessing import Process

import cv2
from vidgear.gears import CamGear, StreamGear

from .config import Config


class StreamingService(Process):
    def __init__(self, config: Config) -> None:
        super().__init__(daemon=True)
        self.config = config
        self._stop_signal = False
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
            "-hls_time": 1,
            "-vcodec": "libx264",
        }

        signal.signal(signal.SIGTERM, self.stop)

    def run(self) -> None:
        self._stream = CamGear(
            source=self.config.DEVICE,
            backend=cv2.CAP_DSHOW,
            logging=self.config.DEBUG,
            **self._camera_params
        ).start()

        self._stream_params["-input_framerate"] = self._stream.framerate
        self._streamer = StreamGear(
            output=self.config.m3u8_file_path,
            format="hls",
            custom_ffmpeg=self.config.FFMPEG_PATH,
            logging=self.config.DEBUG,
            **self._stream_params
        )

        try:
            while not self._stop_signal:
                # read frames from stream
                frame = self._stream.read()
                if frame is None:
                    break

                self._streamer.stream(frame)
        finally:
            self._stop_signal = True
            self._stop()

        logging.log(logging.INFO, "Finished!")

    def stop(self):
        self._stop_signal = True

    def _stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()

        if self._streamer is not None:
            self._streamer.terminate()
