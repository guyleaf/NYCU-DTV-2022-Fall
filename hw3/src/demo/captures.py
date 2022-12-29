import threading
import time
from queue import Queue

import cv2


class BaseCapture:
    def __init__(self) -> None:
        pass

    def running(self):
        raise NotImplementedError()

    def start(self):
        raise NotImplementedError()

    def stop(self):
        raise NotImplementedError()

    def read(self) -> cv2.Mat:
        raise NotImplementedError()


class WebCamCapture(BaseCapture):
    def __init__(self, source: int, transform=None) -> None:
        if isinstance(source, int):
            self._source = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            self._source.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
            self._source.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
            if not self._source.isOpened():
                raise ValueError("Unable to open the webcam.")
        else:
            raise ValueError("The source should be a webcam.")

        self._transform = transform
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._capture)
        self._thread.daemon = True

        _, self._frame = self._source.read()
        if self._transform is not None:
            self._frame = self._transform(self._frame)

    def _capture(self):
        # keep looping infinitely
        while not self._stop_event.is_set():
            # read the next frame from the file
            (grabbed, frame) = self._source.read()

            # if the `grabbed` boolean is `False`, then we have
            # reached the end of the video file
            if not grabbed:
                self._stop_event.set()
                break

            if self._transform is not None:
                frame = self._transform(frame)

            self._frame = frame

        self._source.release()

    def running(self):
        return not self._stop_event.is_set()

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop_event.set()
        self._thread.join()

    def read(self) -> cv2.Mat:
        return self._frame


class VideoCapture(BaseCapture):
    def __init__(
        self, source: str, queue_size: int = 128, transform=None
    ) -> None:
        if isinstance(source, str):
            self._source = cv2.VideoCapture(source)
            if not self._source.isOpened():
                raise ValueError("Unable to open the video")
        else:
            raise ValueError("The source should be a video.")

        self._transform = transform
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._capture)
        self._thread.daemon = True
        self._queue = Queue(maxsize=queue_size)

    def _capture(self):
        # keep looping infinitely
        while not self._stop_event.is_set():
            # otherwise, ensure the queue has room in it
            if not self._queue.full():
                # read the next frame from the file
                (grabbed, frame) = self._source.read()

                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self._stop_event.set()
                    break

                if self._transform is not None:
                    frame = self._transform(frame)

                # add the frame to the queue
                self._queue.put_nowait(frame)
            else:
                time.sleep(0.1)

        self._source.release()

    def running(self):
        return not self._queue.empty() or not self._stop_event.is_set()

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop_event.set()
        self._thread.join()

    def read(self) -> cv2.Mat:
        frame = self._queue.get(timeout=30)
        self._queue.task_done()
        return frame
