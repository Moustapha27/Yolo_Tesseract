from kedro.pipeline import Pipeline, node, pipeline
from .nodes import configure_tesseract, prepare_ocr_data, perform_ocr

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=configure_tesseract,
                inputs="params:tesseract",
                outputs=None,
                name="configure_tesseract_node"
            ),
            node(
                func=prepare_ocr_data,
                inputs=["yolo_predictions", "params:ocr.output_dir"],
                outputs="ocr_input_data",
                name="prepare_ocr_data_node"
            ),
            node(
                func=perform_ocr,
                inputs=["ocr_input_data", "params:ocr.lang"],
                outputs="ocr_results",
                name="perform_ocr_node"
            )
        ]
    )