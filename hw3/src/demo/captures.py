from math import trunc
from typing import Optional, Union

import cv2


class VideoCapture:
    def __init__(self, source: Union[int, str]) -> None:
        self._source = cv2.VideoCapture(source)
        if not self._source.isOpened():
            raise ValueError("Unable to open video source")

        self._width = trunc(self._source.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = trunc(self._source.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def reset(self) -> bool:
        return self._source.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)

    def get_frame(self) -> Optional[cv2.Mat]:
        ret, frame = self._source.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    # Release the video source when the object is destroyed
    def __del__(self):
        if self._source.isOpened():
            self._source.release()
