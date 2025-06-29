from kedro.pipeline import Pipeline, node, pipeline
from .nodes import load_models, predict_with_yolo, extract_and_recognize_text

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_models,
                inputs="params:yolo.model_path",
                outputs="models",
                name="load_models_node"
            ),
            node(
                func=predict_with_yolo,
                inputs=["models", "params:prediction.image_paths"],
                outputs="yolo_predictions",
                name="predict_with_yolo_node"
            ),
            node(
                func=extract_and_recognize_text,
                inputs=["models", "yolo_predictions"],
                outputs="combined_predictions",
                name="extract_and_recognize_text_node"
            )
        ]
    )