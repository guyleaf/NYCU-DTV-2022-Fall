import os
import sys
import tkinter
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
import tkinter.ttk as ttk
import traceback
from tkinter import CENTER, NSEW, Tk
from types import TracebackType
from typing import Any, Dict, Set, Union

import motmetrics as mm
import numpy as np
import sacred
import torch
import yaml
from PIL import Image, ImageTk

from demo.stream_tracker import StreamOutputPipeline, StreamTrackerManager
from demo.utils import SUPPORTED_VIDEO_EXTENSIONS, is_video_file
from trackformer.datasets.coco import make_coco_transforms
from trackformer.datasets.transforms import Compose
from trackformer.models import build_model
from trackformer.models.tracker import Tracker
from trackformer.util.misc import autopath, nested_dict_to_namespace

mm.lap.default_solver = "lap"


class AppViewModel:
    def __init__(
        self,
        stream_tracker_manager: StreamTrackerManager,
        stream_output_pipeline: StreamOutputPipeline,
        upper_buffer_count: int = 60,
        lower_buffer_count: int = 30
    ) -> None:
        self._stream_tracker_manager = stream_tracker_manager
        self._stream_output_pipeline = stream_output_pipeline
        self._upper_buffer_count = upper_buffer_count
        self._lower_buffer_count = lower_buffer_count

        self._source = None
        self._buffer: list[dict] = []

        self._frame_index = 0
        self._is_camera_source = False
        self._is_buffered = False
        self._filters = set()

    @property
    def filters(self) -> Set[int]:
        return self._filters.copy()

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

        self._stream_tracker_manager.stop_tracker()
        self._stream_tracker_manager.start_tracker(
            source, self._stream_output_pipeline.input_queue
        )
        self._buffer.clear()
        self._frame_index = 0
        return True

    def restart(self):
        self._frame_index = 0

    def retrieve_output_pipeline(self):
        queue = self._stream_output_pipeline.output_queue
        if not queue.empty():
            self._buffer.append(queue.get_nowait())
            queue.task_done()

    def get_frame(self):
        self.retrieve_output_pipeline()

        count = len(self._buffer)
        if count == 0:
            return None

        if self._is_camera_source:
            frame = self._buffer[-1]
            # prevent the buffer from OOM, only keep one frame
            self._buffer = [frame]
        else:
            diff = count - self._frame_index
            if diff >= self._upper_buffer_count:
                self._is_buffered = True
            elif diff < self._lower_buffer_count:
                self._is_buffered = False

            self._frame_index = min(self._frame_index, count - 1)
            frame = self._buffer[self._frame_index]

            if self._is_buffered:
                self._frame_index += 1
            # print(diff)
        return frame


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
        if source:
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
        self._image_container = self._canvas.create_image(0, 0, anchor=CENTER)
        self._canvas.grid(row=0, column=0, sticky=NSEW)

        # TODO: make adjustable
        self._delay = delay
        self._image = None
        self.update()

    def update(self) -> None:
        frame = self._app_view_model.get_frame()
        if frame is not None:
            canvas_width = self._canvas.winfo_width()
            canvas_height = self._canvas.winfo_height()

            frame = Image.fromarray(frame)
            frame.thumbnail(size=(canvas_width, canvas_height))
            self._image = ImageTk.PhotoImage(frame)
            self._canvas.itemconfigure(
                self._image_container, image=self._image
            )
            self._canvas.coords(
                self._image_container, canvas_width // 2, canvas_height // 2
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


def excepthook(
    exctype: type,
    excvalue: BaseException,
    tb: Union[TracebackType, None],
):
    """Display exception in a dialog box."""
    if exctype is not KeyboardInterrupt:
        msg = (
            "An uncaught exception has occurred!\n\n"
            + "\n".join(traceback.format_exception(exctype, excvalue, tb))
            + "\nTerminating."
        )
        messagebox.showerror("Error!", msg)
    sys.exit()


sys.excepthook = excepthook

ex = sacred.Experiment("track")
ex.add_config("cfgs/demo.yaml")
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
    _run,
):
    sacred.commands.print_config(_run)

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

    interpolate = interpolate
    transform = Compose(
        make_coco_transforms("val", img_transform, overflow_boxes=True)
    )
    tracker = Tracker(
        obj_detector,
        obj_detector_post,
        tracker_cfg,
        generate_attention_maps,
        track_logger,
        verbose,
    )

    #############################
    # Initialize GUI and others #
    #############################

    stream_manager = StreamTrackerManager(
        tracker,
        transform,
        interpolate,
        _log,
    )
    stream_output_pipeline = StreamOutputPipeline(
        write_images, generate_attention_maps, obj_detector.num_queries, _log
    )
    stream_output_pipeline.setDaemon(True)
    stream_output_pipeline.start()

    app_view_model = AppViewModel(stream_manager, stream_output_pipeline)

    print("tkinter version:", tkinter.TkVersion)
    root = Tk()
    root.report_callback_exception = excepthook
    app = App(root, app_view_model, delay=1000 // 30)
    app.start()
