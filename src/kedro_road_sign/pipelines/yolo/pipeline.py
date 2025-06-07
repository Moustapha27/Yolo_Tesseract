"""
This is a boilerplate pipeline 'yolo'
generated using Kedro 0.19.12
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_yolov8_model, save_yolov8_model, evaluate_yolov8_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_yolov8_model,
            inputs="yolo",
            outputs="yolov8_model",
            name="train_yolov8_model_node"
        ),
        node(
            func=save_yolov8_model,
            inputs=dict(
                model="yolov8_model",
                filepath="params:yolov8_model_path"
            ),
            outputs=None,
            name="save_yolov8_model_node"
        ),
        node(
            func=evaluate_yolov8_model,
            inputs=dict(
                model="yolov8_model",
                data_yaml="yolo"
            ),
            outputs="yolov8_metrics",
            name="evaluate_yolov8_model_node"
        ),
    ])