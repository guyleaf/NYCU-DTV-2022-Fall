import logging
import os
from typing import Literal

from tap import Tap

from live_streaming_server import Config, LiveStreamingServer, get_root
from live_streaming_server.models import MODELS, YOLOV7_OPENVINO
from live_streaming_server.streaming import StreamingService


class ArgumentParse(Tap):
    server_debug: bool = False  # Enable server debug mode (hot reload)
    streaming_debug: bool = False  # Enable server debug mode (hot reload)
    show_model_output: bool = False

    static_folder: str = os.path.join(
        get_root(), "static"
    )  # Where to put static fules
    m3u8_file: str = "output.m3u8"  # M3U8 filename
    camera_device: int = 0  # Camera device id
    ffmpeg_path: str = (
        "C:\\ProgramData\\chocolatey\\bin\\ffmpeg.exe"  # Custom ffmpeg path
    )

    model: MODELS
    """Path to an .xml or .onnx file with a trained model."""

    infer_device: Literal["CPU", "GPU"] = "CPU"  # Device name for inference.
    pre_api: bool = False  # Use preprocessing api.
    grid: bool = False  # With grid in model.
    end2end: bool = False  # With end2end in model.
    nireq: int = 1  # Number of InferRequests, only work in CPU device.
    """
    Explanation:
    https://docs.openvino.ai/2019_R1/_docs_IE_DG_Intro_to_Performance.html

    Reference:
    https://docs.openvino.ai/2022.3/api/ie_python_api/_autosummary/openvino.runtime.AsyncInferQueue.html
    """


class DemoConfig(Config):
    def from_parser(self, parser: ArgumentParse):
        config.STATIC_FOLDER = parser.static_folder
        config.M3U8_FILE = parser.m3u8_file
        config.DEVICE = parser.camera_device
        config.FFMPEG_PATH = parser.ffmpeg_path
        config.DEBUG = parser.server_debug
        config.STREAMING_DEBUG = parser.streaming_debug
        config.SHOW_MODEL_OUTPUT = parser.show_model_output


if __name__ == "__main__":
    parser = ArgumentParse().parse_args()

    config = DemoConfig()
    config.from_parser(parser)

    if not os.path.exists(config.m3u8_root_dir):
        os.mkdir(config.m3u8_root_dir)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    model = YOLOV7_OPENVINO(
        parser.model,
        parser.infer_device,
        parser.pre_api,
        1,
        parser.nireq,
        parser.grid,
        parser.end2end,
    )
    service = StreamingService(model, config=config)
    server = LiveStreamingServer(service, config=config)
    server.start()
