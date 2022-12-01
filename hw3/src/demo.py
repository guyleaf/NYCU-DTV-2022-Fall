import os
from queue import Queue
import tkinter
import tkinter.ttk as ttk
from tkinter import NS, NSEW, Tk
from typing import Any, Dict, Union

import motmetrics as mm
import numpy as np
import sacred
import torch
import yaml
from demo.stream_tracker import StreamTracker

from trackformer.datasets.coco import make_coco_transforms
from trackformer.datasets.transforms import Compose
from trackformer.models import build_model
from trackformer.models.tracker import Tracker
from trackformer.util.misc import autopath, nested_dict_to_namespace

mm.lap.default_solver = "lap"


class StreamManager:
    def __init__(
        self,
        seed: int,
        obj_detect_checkpoint_file: str,
        tracker_cfg: Dict[str, Any],
        interpolate: bool,
        verbose: bool,
        generate_attention_maps: bool,
        _log,
    ) -> None:
        # set all seeds
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            torch.backends.cudnn.deterministic = True

        ##########################
        # Initialize the modules #
        ##########################

        obj_detect_config_path = os.path.join(
            os.path.dirname(obj_detect_checkpoint_file), "config.yaml"
        )
        obj_detect_args = nested_dict_to_namespace(
            yaml.unsafe_load(open(obj_detect_config_path))
        )
        img_transform = obj_detect_args.img_transform
        obj_detector, _, obj_detector_post = build_model(obj_detect_args)

        obj_detect_checkpoint = torch.load(
            obj_detect_checkpoint_file,
            map_location=lambda storage, loc: storage,
        )

        obj_detect_state_dict = obj_detect_checkpoint["model"]
        # obj_detect_state_dict = {
        #     k: obj_detect_state_dict[k] if k in obj_detect_state_dict
        #     else v
        #     for k, v in obj_detector.state_dict().items()}

        obj_detect_state_dict = {
            k.replace("detr.", ""): v
            for k, v in obj_detect_state_dict.items()
            if "track_encoding" not in k
        }

        obj_detector.load_state_dict(obj_detect_state_dict)
        if "epoch" in obj_detect_checkpoint:
            _log.info(
                f"INIT object detector [EPOCH: {obj_detect_checkpoint['epoch']}]"
            )

        obj_detector.cuda()

        if hasattr(obj_detector, "tracking"):
            obj_detector.tracking()

        track_logger = None
        if verbose:
            track_logger = _log.info

        self._log = _log
        self._interpolate = interpolate
        self._transform = Compose(
            make_coco_transforms("val", img_transform, overflow_boxes=True)
        )
        self._tracker = Tracker(
            obj_detector,
            obj_detector_post,
            tracker_cfg,
            generate_attention_maps,
            track_logger,
            verbose,
        )
        self._stream_trackers: Dict[int, StreamTracker] = {}
        self._count = 0

    def stop_all(self):
        for id in self._stream_trackers.keys():
            self.stop_tracker(id)

    def stop_tracker(self, id: int):
        if id in self._stream_trackers:
            stream_tracker = self._stream_trackers.pop(id)
            stream_tracker.stop()
            stream_tracker.join()
            del stream_tracker

    def start_tracker(self, data_source: Union[int, str]):
        queue = Queue()
        stream_tracker = StreamTracker(
            self._tracker,
            data_source,
            self._transform,
            queue,
            self._interpolate,
            self._log,
        )
        stream_tracker.start()
        self._stream_trackers[self._count] = stream_tracker
        self._count += 1
        return self._count - 1, queue


class DemoSettings:
    def __init__(self, content: ttk.Frame) -> None:
        self._content = content
        self._content.columnconfigure(0, weight=1)

        sources = ttk.Labelframe(
            self._content, text="Sources", padding=(5, 0, 5, 5)
        )
        sources.columnconfigure(0, weight=1)
        sources.rowconfigure((0, 1), minsize=30)
        sources.grid(row=0, column=0, sticky=NSEW)

        self._camera_button = ttk.Button(sources, text="Camera")
        self._camera_button.grid(row=0, column=0, sticky=NSEW, pady=(0, 1.5))
        self._video_button = ttk.Button(sources, text="Video")
        self._video_button.grid(row=1, column=0, sticky=NSEW, pady=(1.5, 0))


class TrackingVisualizer:
    def __init__(
        self,
        content: ttk.Frame,
        delay: int,
        stream_manager: StreamManager,
    ) -> None:
        self._content = content
        self._content.columnconfigure(0, weight=1)
        self._content.rowconfigure(0, weight=1)

        self._canvas = tkinter.Canvas(content, background="gray50")
        self._canvas.grid(row=0, column=0, sticky=NSEW)

        # TODO: make adjustable
        self._delay = delay
        self._stream_manager = stream_manager

    def set_data_source(self):
        pass

    def update(self) -> None:
        self._canvas.after(self._delay, self.update)


class App:
    def __init__(
        self,
        window: Tk,
        title: str = "MOT Demo",
        delay: int = 1000 // 30,
        stream_manager: StreamManager = None,
    ) -> None:
        self._window = window
        self._window.geometry("1000x600")
        self._window.title(title)
        self._window.columnconfigure(0, weight=1)
        self._window.rowconfigure(0, weight=1)

        self._content = ttk.Frame(self._window, padding=(3, 3, 3, 3))
        self._content.columnconfigure(0, weight=1)
        self._content.columnconfigure(2, weight=2)
        self._content.rowconfigure(0, weight=1)

        bar = ttk.Frame(self._content)
        seperator = ttk.Separator(self._content, orient=tkinter.VERTICAL)
        visualizer = ttk.Frame(self._content)

        visualizer.grid(row=0, column=2, sticky=NSEW)
        seperator.grid(row=0, column=1, sticky=NSEW)
        bar.grid(row=0, column=0, sticky=NSEW)
        self._content.grid(row=0, column=0, sticky=NSEW)

        self._bar = DemoSettings(bar)

        if stream_manager is not None:
            self._visualizer = TrackingVisualizer(
                visualizer, delay, stream_manager
            )

    def start(self) -> None:
        self._window.mainloop()


ex = sacred.Experiment("track")
ex.add_config("cfgs/track.yaml")
ex.add_named_config("reid", "cfgs/track_reid.yaml")


@ex.automain
@autopath()
def main(
    seed: int,
    obj_detect_checkpoint_file: str,
    tracker_cfg: Dict[str, Any],
    interpolate: bool,
    verbose: bool,
    generate_attention_maps: bool,
    _log,
):
    stream_manager = StreamManager(
        seed,
        obj_detect_checkpoint_file,
        tracker_cfg,
        interpolate,
        verbose,
        generate_attention_maps,
    )
    print("tkinter version:", tkinter.TkVersion)

    root = Tk()
    app = App(root, stream_manager=stream_manager)
    app.start()
