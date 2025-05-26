"""
This is a boilerplate pipeline 'yolo'
generated using Kedro 0.19.12
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_yolov8_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_yolov8_model,
            inputs="yolo",
            outputs="yolov8_model",
            name="train_yolov8_model_node"
        )
    ])