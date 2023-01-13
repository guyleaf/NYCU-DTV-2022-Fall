from multiprocessing.connection import Connection
from flask import Blueprint, render_template, request, session, current_app
import secrets
import os

from .utils import make_response


web = Blueprint("web", __name__)
adding_streamer_pipe: Connection = None
updating_classes_pipe: Connection = None


def assign_pipes(streamer_pipe: Connection, classes_pipe: Connection):
    global adding_streamer_pipe, updating_classes_pipe
    adding_streamer_pipe = streamer_pipe
    updating_classes_pipe = classes_pipe


@web.route("/", methods=["GET"])
def index():
    if "id" not in session:
        session["id"] = secrets.token_hex(16)
        session["selected_classes"] = {}
        adding_streamer_pipe.send((session["id"],))

    return render_template(
        "index.html",
        classes=current_app.config["CLASSES"],
        id=session["id"],
        selected_classes=session["selected_classes"],
    )


@web.route("/check", methods=["GET"])
def check():
    if "id" not in session:
        return make_response("Please refresh the website.", status_code=400)

    config = current_app.config
    m3u8_file_path = os.path.join(
        config["HLS_ROOT_FOLDER"], session["id"], config["M3U8_FILE"]
    )
    if os.path.exists(m3u8_file_path):
        return make_response("M3U8 file is created.", data={"exists": True})
    return make_response(
        "M3U8 file is not created yet.", data={"exists": False}
    )


@web.route("/detections", methods=["POST"])
def detections():
    if "id" not in session:
        return make_response("Please refresh the website.", status_code=400)

    data = request.get_json()
    if data is not None:
        updating_classes_pipe.send((session["id"], data))
        session["selected_classes"] = data
        return make_response("Success")
    return make_response("Incorrect data format.", status_code=400)



