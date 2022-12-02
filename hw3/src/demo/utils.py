import mimetypes

mimetypes.init()


SUPPORTED_VIDEO_EXTENSIONS = [("Video Files", ("*.mp4", "*.mkv"))]


def is_video_file(path: str):
    extension = mimetypes.guess_type(path)[0]
    return extension is not None and "video" in extension


if __name__ == "__main__":
    test = is_video_file("test.mp4")
    print(test)
