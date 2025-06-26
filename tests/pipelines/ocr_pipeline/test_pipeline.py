import cv2
import numpy as np
import pytest
import os

from src.kedro_road_sign.pipelines.ocr_pipeline.nodes import run_ocr_on_detection
from kedro.pipeline import Pipeline
from src.kedro_road_sign.pipelines.ocr_pipeline.pipeline import create_pipeline


@pytest.fixture
def sample_images(tmp_path):
    # Crée une image temporaire avec du texte
    image = np.ones((100, 300), dtype=np.uint8) * 255  # fond blanc
    cv2.putText(image, 'Hello', (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0), 3)
    
    png_path = tmp_path / "test_ocr.png"
    jpg_path = tmp_path / "test_ocr.jpg"
    
    cv2.imwrite(str(png_path), image)
    cv2.imwrite(str(jpg_path), image)
    
    return tmp_path

def test_run_ocr_on_detection(sample_images):
    results = run_ocr_on_detection(str(sample_images))
    assert "test_ocr.png" in results
    assert "test_ocr.jpg" in results
    assert isinstance(results["test_ocr.png"], str)
    assert isinstance(results["test_ocr.jpg"], str)
    assert "hello" in results["test_ocr.png"].lower()
    assert "hello" in results["test_ocr.jpg"].lower()

def test_pipeline_structure():
    pipeline = create_pipeline()
    assert isinstance(pipeline, Pipeline)
    node_names = [n.name for n in pipeline.nodes]
    assert "ocr_node" in node_names
