import mimetypes
from typing import Optional

import cv2

mimetypes.init()


SUPPORTED_VIDEO_EXTENSIONS = [("Video Files", ("*.mp4", "*.mkv"))]


def is_video_file(path: str):
    extension = mimetypes.guess_type(path)[0]
    return extension is not None and "video" in extension


def cv2_video_transform(frame: cv2.Mat):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def cv2_webcam_transform(frame: cv2.Mat):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)
    return frame


def get_track_id_by_xy(tracks: dict, x: int, y: int) -> Optional[int]:
    for track_id, track in tracks.items():
        x1, y1, x2, y2 = track["bbox"]
        if x1 <= x and x <= x2 and y1 <= y and y <= y2:
            return track_id
    return None


if __name__ == "__main__":
    test = is_video_file("test.mp4")
    print(test)
