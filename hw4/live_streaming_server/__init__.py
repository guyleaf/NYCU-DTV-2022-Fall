from flask import Flask

from .streaming import StreamingService
from .web import web


class LiveStreamingServer:
    def __init__(
        self,
        streaming_service: StreamingService,
        host: str = "0.0.0.0",
        port: int = 8080,
        debug: bool = False,
    ) -> None:
        self._streaming_service = streaming_service
        self._host = host
        self._port = port
        self._debug = debug
        self._app = Flask(__name__)
        self._app.register_blueprint(web)

    def start(self):
        self._app.run(host=self._host, port=self._port, debug=self._debug)
