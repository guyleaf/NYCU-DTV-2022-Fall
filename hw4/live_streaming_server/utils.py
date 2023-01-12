import platform
from typing import Any

CURRENT_PLATFORM = platform.system()


def is_darwin():
    return CURRENT_PLATFORM == "Darwin"


def is_windows():
    return CURRENT_PLATFORM == "Windows"


def is_linux():
    return CURRENT_PLATFORM == "Linux"


def make_response(message: str, data: Any = None, status_code: int = 200):
    return {"message": message, "data": data}, status_code
