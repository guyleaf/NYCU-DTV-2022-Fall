import os
import tkinter
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
import tkinter.ttk as ttk
from queue import Queue
from tkinter import NSEW, Tk
from typing import Any, Dict, Set, Union

import motmetrics as mm
import numpy as np
import sacred
import torch
import yaml
from PIL import Image, ImageTk

from demo.stream_tracker import StreamTracker
from demo.utils import SUPPORTED_VIDEO_EXTENSIONS, is_video_file
from trackformer.datasets.coco import make_coco_transforms
from trackformer.datasets.transforms import Compose
from trackformer.models import build_model
from trackformer.models.tracker import Tracker
from trackformer.util.misc import autopath, nested_dict_to_namespace
from trackformer.util.track_utils import plot_single_frame

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


class AppViewModel:
    def __init__(
        self,
        stream_manager: StreamManager,
        write_images: Union[bool, str],
        generate_attention_maps: bool,
    ) -> None:
        self._stream_manager = stream_manager
        self._write_images = write_images
        self._generate_attention_maps = generate_attention_maps

        self._source = None
        self._buffer: list[dict] = []
        self._queue = None

        self._frame_index = 0
        self._is_camera_source = False
        self._filters = set()

    @property
    def filters(self) -> Set[int]:
        return self._filters.copy()

    def _post_process(self, result: dict):
        image = result["image"]
        tracks = result["tracks"].copy()

        for track_id in self._filters:
            if track_id in tracks:
                del tracks[track_id]

        image = plot_single_frame(
            tracks, image, self._write_images, self._generate_attention_maps
        )
        return image

    def add_track_id_to_filters(self, id: int):
        self._filters.add(id)

    def remove_track_id_from_filters(self, id: int):
        self._filters.discard(id)

    def play(self, source: Union[int, str]) -> bool:
        # TODO: check the source is available?
        self._is_camera_source = isinstance(source, int)
        if not self._is_camera_source:
            if not os.path.isfile(source) or not is_video_file(source):
                return False

        # TODO: whether we really need to support tracking multiple videos?
        self._stream_manager.stop_all()
        del self._queue

        _, self._queue = self._stream_manager.start_tracker(source)
        self._buffer.clear()
        return True

    def restart(self):
        self._frame_index = 0

    def get_frame(self):
        if not self._queue.empty():
            self._buffer.append(self._queue.get(block=False))

        count = len(self._buffer)
        if count == 0:
            return None

        if self._is_camera_source:
            # prevent the buffer from OOM, only keep one frame
            self._buffer = self._buffer[-1:]
            result = self._buffer[-1]
        else:
            if self._frame_index >= count:
                self._frame_index = count - 1

            result = self._buffer[self._frame_index]
            self._frame_index += 1
        return self._post_process(result)


class DemoSettings:
    def __init__(
        self, content: ttk.Frame, app_view_model: AppViewModel
    ) -> None:
        self._app_view_model = app_view_model

        self._content = content
        self._content.columnconfigure(0, weight=1)

        sources = ttk.Labelframe(
            self._content, text="Sources", padding=(5, 0, 5, 5)
        )
        sources.columnconfigure(0, weight=1)
        sources.rowconfigure((0, 1), minsize=30)
        sources.grid(row=0, column=0, sticky=NSEW)

        self._camera_button = ttk.Button(
            sources, text="Camera", command=self.on_camera_button_clicked
        )
        self._camera_button.grid(row=0, column=0, sticky=NSEW, pady=(0, 1.5))
        self._video_button = ttk.Button(
            sources, text="Video", command=self.on_video_button_clicked
        )
        self._video_button.grid(row=1, column=0, sticky=NSEW, pady=(1.5, 0))

    def on_camera_button_clicked(self):
        if not self._app_view_model.play(0):
            messagebox.showerror("Camera", "Cannot open the camera.")

    def on_video_button_clicked(self):
        source = filedialog.askopenfilename(
            filetypes=SUPPORTED_VIDEO_EXTENSIONS
        )
        if not self._app_view_model.play(source):
            messagebox.showerror("Video", "Cannot open the video.")


class TrackingVisualizer:
    def __init__(
        self,
        content: ttk.Frame,
        delay: int,
        app_view_model: AppViewModel,
    ) -> None:
        self._app_view_model = app_view_model

        self._content = content
        self._content.columnconfigure(0, weight=1)
        self._content.rowconfigure(0, weight=1)

        self._canvas = tkinter.Canvas(content, background="gray50")
        self._canvas.grid(row=0, column=0, sticky=NSEW)

        # TODO: make adjustable
        self._delay = delay
        self.update()

    def update(self) -> None:
        frame = self._app_view_model.get_frame()
        if frame is not None:
            frame = Image.fromarray(frame)
            frame = ImageTk.PhotoImage(frame)
            self._canvas.delete("frame")
            self._canvas.create_image(
                0, 0, image=frame, anchor=NSEW, tags=("frame")
            )
        self._canvas.after(self._delay, self.update)


class App:
    def __init__(
        self,
        window: Tk,
        app_view_model: AppViewModel,
        title: str = "MOT Demo",
        delay: int = 1000 // 30,
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

        self._bar = DemoSettings(bar, app_view_model)

        self._visualizer = TrackingVisualizer(
            visualizer, delay, app_view_model
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
    write_images: Union[bool, str],
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
        _log,
    )
    app_view_model = AppViewModel(
        stream_manager, write_images, generate_attention_maps
    )
    print("tkinter version:", tkinter.TkVersion)

    root = Tk()
    app = App(root, app_view_model)
    app.start()
