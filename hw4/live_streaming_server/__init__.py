import atexit
import os

from flask import Flask
from flask_cors import CORS

from .config import Config
from .streaming import StreamingService
from .routes import web


def get_root():
    return os.path.dirname(os.path.abspath(__file__))


class LiveStreamingServer:
    def __init__(
        self, streaming_service: StreamingService, config: Config
    ) -> None:
        self._streaming_service = streaming_service
        self._config = config

        self._app = Flask(__name__, static_folder=config.STATIC_FOLDER)
        CORS(self._app)
        config.configure(self._app)
        self._app.register_blueprint(web)

        atexit.register(self.stop)

    def start(self):
        self._streaming_service.start()
        self._app.run(
            threaded=True, host=self._config.HOST, port=self._config.PORT
        )

    def stop(self):
        print("Stopping server...")
        self._streaming_service.terminate()
        self._streaming_service.join()