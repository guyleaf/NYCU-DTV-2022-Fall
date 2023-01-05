import os

from tap import Tap

from live_streaming_server import Config, LiveStreamingServer, get_root
from live_streaming_server.streaming import StreamingService


class ArgumentParse(Tap):
    static_folder: str = os.path.join(get_root(), "static")
    m3u8_file: str = "output.m3u8"
    device: int = 0  # camera device id
    ffmpeg_path: str = "C:\\ProgramData\\chocolatey\\bin\\ffmpeg.exe"
    debug: bool = False


class DemoConfig(Config):
    def from_parser(self, parser: ArgumentParse):
        config.STATIC_FOLDER = parser.static_folder
        config.M3U8_FILE = parser.m3u8_file
        config.DEVICE = parser.device
        config.FFMPEG_PATH = parser.ffmpeg_path
        config.DEBUG = parser.debug


if __name__ == "__main__":
    parser = ArgumentParse().parse_args()

    config = DemoConfig()
    config.from_parser(parser)

    if not os.path.exists(config.m3u8_root_dir):
        os.mkdir(config.m3u8_root_dir)

    service = StreamingService(config=config)
    server = LiveStreamingServer(service, config=config)
    server.start()
