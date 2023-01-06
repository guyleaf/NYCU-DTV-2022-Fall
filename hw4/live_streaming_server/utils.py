import platform

CURRENT_PLATFORM = platform.system()


def is_darwin():
    return CURRENT_PLATFORM == "Darwin"


def is_windows():
    return CURRENT_PLATFORM == "Windows"


def is_linux():
    return CURRENT_PLATFORM == "Linux"
