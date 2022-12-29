from flask import Blueprint


web = Blueprint("web", __name__)


@web.route("/")
def hello_world():
    return "<p>Hello, World!</p>"
