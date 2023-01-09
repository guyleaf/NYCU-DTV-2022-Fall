import os
import secrets

from flask import Flask


class Config:
    # Web server
    DEBUG: bool = False
    SECRET_KEY: str = secrets.token_hex()
    STATIC_FOLDER: str = "static"

    HOST: str = "0.0.0.0"
    PORT: int = 8080

    # Streaming Service
    STREAMING_DEBUG: bool = False
    DEVICE: int = 0
    M3U8_FILE: str = "output.m3u8"
    FFMPEG_PATH: str = "C:\\ProgramData\\chocolatey\\bin\\ffmpeg.exe"
    SHOW_MODEL_OUTPUT: bool = False

    @property
    def hls_root_folder(self):
        return os.path.join(self.STATIC_FOLDER, "live")

    @property
    def m3u8_file_path(self):
        return os.path.join(self.hls_root_folder, self.M3U8_FILE)

    def configure(self, app: Flask):
        app.config.from_object(self)
        app.config.setdefault("HLS_ROOT_FOLDER", self.hls_root_folder)
