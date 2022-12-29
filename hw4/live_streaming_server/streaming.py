from multiprocessing import Process


class StreamingService(Process):
    def __init__(self, hls_folder: str) -> None:
        super().__init__(daemon=True)

    def run(self) -> None:
        return super().run()
