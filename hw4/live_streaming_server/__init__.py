import atexit
import logging
import os

from multiprocessing.connection import Connection
from flask import Flask
from flask_cors import CORS

from .config import Config
from .streaming import StreamingService
from .routes import web, assign_pipes


def get_root():
    return os.path.dirname(os.path.abspath(__file__))


class LiveStreamingServer:
    def __init__(
        self,
        adding_streamer_pipe: Connection,
        updating_classes_pipe: Connection,
        streaming_service: StreamingService,
        config: Config,
    ) -> None:
        self._streaming_service = streaming_service
        self._config = config

        assign_pipes(adding_streamer_pipe, updating_classes_pipe)

        self._app = Flask(__name__, static_folder=config.STATIC_FOLDER)
        self._config.configure(self._app)
        CORS(self._app)
        self._app.register_blueprint(web)

        atexit.register(self.stop)

    def start(self):
        self._streaming_service.start()
        self._app.run(
            threaded=True, host=self._config.HOST, port=self._config.PORT
        )

    def stop(self):
        logging.log(logging.INFO, "Stopping server...")
        if self._streaming_service.is_alive():
            self._streaming_service.terminate()
            self._streaming_service.join()
