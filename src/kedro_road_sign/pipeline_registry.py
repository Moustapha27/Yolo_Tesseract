from .pipelines.yolo import create_pipeline as yolo
from .pipelines.ocr_pipeline import create_pipeline as ocr_pipeline
from .pipelines.prediction_pipeline import create_pipeline as prediction_pipeline
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Enregistre les pipelines du projet Kedro"""

    yo = yolo()
    ocr = ocr_pipeline()
    pred_pipeline = prediction_pipeline()


    return {
        "__default__": yo + ocr + pred_pipeline,
        "yolo": yo,
        "ocr_pipeline": ocr,
        "prediction_pipeline": pred_pipeline,
    }
