from math import trunc
from typing import Optional, Union

import cv2


class VideoCapture:
    def __init__(self, source: Union[int, str]) -> None:
        if isinstance(source, int):
            self._source = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            self._source.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
            self._source.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
            self._source.set(cv2.CAP_PROP_FPS, 30)
        else:
            self._source = cv2.VideoCapture(source)

        if not self._source.isOpened():
            raise ValueError("Unable to open the video source")

        # self._source.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    def get_frame(self) -> Optional[cv2.Mat]:
        ret, frame = self._source.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    # Release the video source when the object is destroyed
    def __del__(self):
        if self._source.isOpened():
            self._source.release()
