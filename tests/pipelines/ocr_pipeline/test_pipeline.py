import os
import unittest
from unittest.mock import patch, MagicMock

import pytest
from kedro.runner import SequentialRunner
from kedro.pipeline import Pipeline

# Chemins du projet (à adapter si besoin)
from src.kedro_road_sign.pipelines.ocr_pipeline.nodes import run_ocr_on_detection
from src.kedro_road_sign.pipelines.ocr_pipeline.pipeline import create_pipeline


# ---------------------------------------------------
# 1️⃣ Tests unitaires classiques avec unittest
# ---------------------------------------------------

class TestRunOcrOnDetection(unittest.TestCase):

    @patch("os.listdir")
    @patch("cv2.imread")
    @patch("pytesseract.image_to_string")
    def test_run_ocr_on_detection(self, mock_ocr, mock_imread, mock_listdir):
        # Données simulées
        mock_listdir.return_value = ["image1.jpg", "image2.png", "not_image.txt"]
        mock_imread.return_value = MagicMock()  # faux objet image
        mock_ocr.side_effect = ["Texte image1", "Texte image2"]

        folder_path = "/fake/path"
        expected = {
            "image1.jpg": "Texte image1",
            "image2.png": "Texte image2"
        }

        result = run_ocr_on_detection(folder_path)
        self.assertEqual(result, expected)
        mock_listdir.assert_called_once_with(folder_path)
        self.assertEqual(mock_imread.call_count, 2)
        self.assertEqual(mock_ocr.call_count, 2)


# ---------------------------------------------------
# 2️⃣ Test de pipeline Kedro avec pytest
# ---------------------------------------------------

@pytest.fixture
def dummy_input():
    return [
        {"filename": "img1.jpg", "text_region": "Hello"},
        {"filename": "img2.jpg", "text_region": "World"},
    ]

def dummy_ocr_on_detection(images):
    # Fonction simulée remplaçant run_ocr_on_detection
    return {img["filename"]: img["text_region"] for img in images}

def test_ocr_pipeline_execution(monkeypatch, dummy_input):
    from src.kedro_road_sign.pipelines import ocr_pipeline

    monkeypatch.setattr(ocr_pipeline.nodes, "run_ocr_on_detection", dummy_ocr_on_detection)

    # ✅ On appelle bien la fonction
    pipeline = ocr_pipeline.pipeline.create_pipeline()

    # ✅ pipeline est bien un objet Kedro Pipeline
    assert isinstance(pipeline, Pipeline)

    runner = SequentialRunner()

    result = runner.run({
        "params:images_for_ocr": dummy_input
    }, pipeline)

    assert "ocr_results" in result
