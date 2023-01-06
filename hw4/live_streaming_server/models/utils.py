import os


def get_model_path(model: str):
    model = model.replace(".", os.sep) + ".xml"
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), model)
