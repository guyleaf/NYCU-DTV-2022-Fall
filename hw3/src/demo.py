import os
import sys
import threading
import tkinter
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
import tkinter.ttk as ttk
import traceback
import yappi
from tkinter import CENTER, NSEW, Tk
from types import TracebackType
from typing import Any, Dict, Union

import motmetrics as mm
import numpy as np
import sacred
import torch
import yaml
from PIL import Image, ImageTk
from demo.captures import VideoCapture, WebCamCapture

from demo.stream_tracker import StreamOutputPipeline, StreamTrackerManager
from demo.utils import (
    SUPPORTED_VIDEO_EXTENSIONS,
    cv2_video_transform,
    cv2_webcam_transform,
    get_track_id_by_xy,
)
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
    ) -> None:
        self._stream_tracker_manager = stream_tracker_manager
        self._stream_output_pipeline = stream_output_pipeline

        self._source = None
        self._buffer: list[dict] = []

        self._frame_index = 0
        self._is_camera_source = False
        self._filters = set()

    def _clear_all_filters(self):
        self._filters = set()
        self._stream_output_pipeline.input_filters.put(set())

    def _update_filters(self, tracks: dict, x: int, y: int) -> None:
        track_id = get_track_id_by_xy(tracks, x, y)
        if track_id is None:
            return

        if track_id in self._filters:
            self._filters.discard(track_id)
        else:
            self._filters.add(track_id)
        self._stream_output_pipeline.input_filters.put(self._filters.copy())

    def _get_current_tracks(self) -> None:
        if self._is_camera_source:
            tracks = self._buffer[-1]["tracks"]
            orig_size = self._buffer[-1]["orig_size"]
        else:
            tracks = self._buffer[self._frame_index - 1]["tracks"]
            orig_size = self._buffer[self._frame_index - 1]["orig_size"]
        return tracks, orig_size

    def update_filters(
        self, x: int, y: int, image_size: "tuple[int, int]"
    ) -> None:
        tracks, orig_size = self._get_current_tracks()

        # convert x, y into original image space
        x = int(x * (orig_size[1] / image_size[1]))
        y = int(y * (orig_size[0] / image_size[0]))

        threading.Thread(
            target=self._update_filters,
            args=(tracks, x, y),
        ).start()

    def select_all_tracks(self) -> None:
        self._clear_all_filters()
        self._stream_output_pipeline.select_default = True

    def hide_all_tracks(self) -> None:
        self._clear_all_filters()
        self._stream_output_pipeline.select_default = False

    def play(self, source: Union[int, str]) -> bool:
        self._is_camera_source = isinstance(source, int)
        try:
            if self._is_camera_source:
                capture = WebCamCapture(source, transform=cv2_webcam_transform)
            else:
                capture = VideoCapture(source, transform=cv2_video_transform)
        except BaseException:
            return False

        self._stream_tracker_manager.stop_tracker()
        self._stream_tracker_manager.start_tracker(
            capture, self._stream_output_pipeline.input_queue
        )
        self._buffer = []
        self._filters = set()
        self._frame_index = 0
        self._stream_output_pipeline.input_filters.put_nowait(set())
        return True

    def restart(self) -> None:
        self._frame_index = 0

    def get_frame(self) -> np.ndarray:
        queue = self._stream_output_pipeline.output_queue
        if not queue.empty():
            self._buffer.append(queue.get_nowait())
            queue.task_done()

        count = len(self._buffer)
        if count == 0:
            return None

        if self._is_camera_source:
            frame = self._buffer[-1]
            # prevent the buffer from OOM, only keep one frame
            self._buffer = [frame]
            frame = frame["image"]
        else:
            self._frame_index = min(self._frame_index, count - 1)
            frame = self._buffer[self._frame_index]["image"]
            self._frame_index += 1
        return frame


class DemoSettings:
    def __init__(
        self, content: ttk.Frame, app_view_model: AppViewModel
    ) -> None:
        self._app_view_model = app_view_model
        self._content = content

        sources = ttk.Labelframe(
            self._content, text="Sources", padding=(5, 0, 5, 5)
        )

        self._camera_button = ttk.Button(
            sources, text="Camera", command=self.on_camera_button_clicked
        )
        self._camera_button.grid(row=0, column=0, sticky=NSEW, padx=(0, 1.5))
        self._video_button = ttk.Button(
            sources, text="Video", command=self.on_video_button_clicked
        )
        self._video_button.grid(row=0, column=1, sticky=NSEW, padx=(1.5, 0))

        sources.columnconfigure(list(range(sources.grid_size()[0])), weight=1)
        sources.rowconfigure(list(range(sources.grid_size()[1])), minsize=30)
        sources.grid(row=0, column=0, sticky=NSEW)

        bbox_control = ttk.Labelframe(
            self._content, text="BBox Control", padding=(5, 0, 5, 5)
        )

        # FIXME: restarting the video will not have bbox selection function
        # because we use StreamOutputPipeline to filter out the bboxes
        # self._video_restart_button = ttk.Button(
        #     controls, text="Restart", command=self._app_view_model.restart
        # )
        # self._video_restart_button.grid(row=0, column=0, sticky="nse")

        self._select_all_button = ttk.Button(
            bbox_control,
            text="Select All",
            command=self._app_view_model.select_all_tracks,
        )
        self._select_all_button.grid(row=0, column=0, sticky=NSEW)
        self._hide_all_button = ttk.Button(
            bbox_control,
            text="Hide All",
            command=self._app_view_model.hide_all_tracks,
        )
        self._hide_all_button.grid(row=0, column=1, sticky=NSEW)

        bbox_control.columnconfigure(
            list(range(bbox_control.grid_size()[0])), weight=1
        )
        bbox_control.rowconfigure(
            list(range(bbox_control.grid_size()[1])), minsize=30
        )
        bbox_control.grid(row=1, column=0, sticky=NSEW)

        self._content.columnconfigure(0, weight=1)

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
        app_view_model: AppViewModel,
    ) -> None:
        self._app_view_model = app_view_model

        self._content = content
        self._content.columnconfigure(0, weight=1)
        self._content.rowconfigure(0, weight=1)

        self._canvas = tkinter.Canvas(content, background="gray50")
        self._image_container = self._canvas.create_image(0, 0, anchor=CENTER)
        self._canvas.tag_bind(
            self._image_container, "<Button-1>", self.click_image
        )
        self._canvas.grid(row=0, column=0, sticky=NSEW)

        # TODO: make adjustable
        self._delay = 1000 // 30
        self.update()

    def click_image(self, event: tkinter.Event) -> None:
        image_size = (self._image.height(), self._image.width())
        x = event.x - (self._canvas_width - image_size[1]) // 2
        y = event.y - (self._canvas_height - image_size[0]) // 2
        self._app_view_model.update_filters(x, y, image_size)

    def update(self) -> None:
        frame = self._app_view_model.get_frame()
        if frame is not None:
            self._canvas_width = self._canvas.winfo_width()
            self._canvas_height = self._canvas.winfo_height()

            frame = Image.fromarray(frame)
            frame.thumbnail(size=(self._canvas_width, self._canvas_height))
            self._image = ImageTk.PhotoImage(frame)
            self._canvas.itemconfigure(
                self._image_container, image=self._image
            )
            self._canvas.coords(
                self._image_container,
                self._canvas_width // 2,
                self._canvas_height // 2,
            )
        else:
            self._canvas.itemconfigure(self._image_container, image=None)
        self._canvas.after(self._delay, self.update)


class App:
    def __init__(
        self, window: Tk, app_view_model: AppViewModel, title: str = "MOT Demo"
    ) -> None:
        self._window = window
        self._window.geometry("1000x600")
        self._window.title(title)
        self._window.columnconfigure(0, weight=1)
        self._window.rowconfigure(0, weight=1)

        self._content = ttk.Frame(self._window)
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
        self._visualizer = TrackingVisualizer(visualizer, app_view_model)

        for child in self._content.winfo_children():
            child.grid_configure(padx=5, pady=5)

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
    sys.exit(1)


sys.excepthook = excepthook

ex = sacred.Experiment("demo")
ex.add_config("cfgs/demo.yaml")
ex.add_named_config("reid", "cfgs/track_reid.yaml")


@ex.automain
@autopath()
def main(
    seed: int,
    obj_detect_checkpoint_file: str,
    tracker_cfg: Dict[str, Any],
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

    print("Clock type:", yappi.get_clock_type())
    yappi.start()

    stream_manager = StreamTrackerManager(
        tracker,
        transform,
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
    app = App(root, app_view_model)
    app.start()

    stream_manager.stop_tracker()
    stream_output_pipeline.stop()
    stream_output_pipeline.join()

    yappi.stop()
    threads = yappi.get_thread_stats()
    with open("debug.log", "w") as f:
        for thread in threads:
            if thread.name in [
                "StreamOutputPipeline",
                "StreamTracker",
                "_MainThread",
            ]:
                print(
                    "Function stats for (%s) (%d)" % (thread.name, thread.id),
                    file=f,
                )  # it is the Thread.__class__.__name__
                yappi.get_func_stats(ctx_id=thread.id).print_all(out=f)
