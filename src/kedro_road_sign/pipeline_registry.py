from .pipelines.yolo import create_pipeline as yolo
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Enregistre les pipelines du projet Kedro"""

    yo = yolo()


    return {
        "__default__": yo,
        "yolo": yo,
    }
