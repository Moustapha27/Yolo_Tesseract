import cv2
import numpy as np
import pytest
import os

from src.kedro_road_sign.pipelines.ocr_pipeline.nodes import run_ocr_on_detection
from kedro.pipeline import Pipeline
from src.kedro_road_sign.pipelines.ocr_pipeline.pipeline import create_pipeline


@pytest.fixture
def sample_image(tmp_path):
    # Crée une image temporaire avec du texte
    image = np.ones((100, 300), dtype=np.uint8) * 255  # fond blanc
    cv2.putText(image, 'Hello', (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0), 3)
    image_path = tmp_path / "test_ocr.png"
    cv2.imwrite(str(image_path), image)
    return tmp_path

def test_run_ocr_on_detection(sample_image):
    results = run_ocr_on_detection(str(sample_image))
    assert "test_ocr.png" in results
    assert isinstance(results["test_ocr.png"], str)
    assert "hello" in results["test_ocr.png"].lower()



def test_pipeline_structure():
    pipeline = create_pipeline()
    assert isinstance(pipeline, Pipeline)
    node_names = [n.name for n in pipeline.nodes]
    assert "ocr_node" in node_names
