from kedro.pipeline import Pipeline, node, pipeline
from .nodes import run_ocr_on_detection

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=run_ocr_on_detection,
            inputs="params:images_for_ocr",
            outputs="ocr_results",
            name="ocr_node",
        ),
    ])
