from kedro.pipeline import Pipeline, node, pipeline
from .nodes import validate_yolo_structure, train_yolov8, evaluate_yolov8

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=validate_yolo_structure,
                inputs="params:yolo.raw_data_path",
                outputs="yolo_data_info",
                name="validate_yolo_structure_node"
            ),
            node(
                func=train_yolov8,
                inputs=["yolo_data_info", "params:yolo.model_output_dir", "params:yolo.epochs", "params:yolo.imgsz"],
                outputs={
                    "best_model": "yolo_best_model",
                    "last_model": "yolo_last_model",
                    "train_results": "yolo_train_results"
                },
                name="train_yolov8_node"
            ),
            node(
                func=evaluate_yolov8,
                inputs=["yolo_best_model", "yolo_data_info"],
                outputs="yolo_metrics",
                name="evaluate_yolov8_node"
            )
        ]
    )