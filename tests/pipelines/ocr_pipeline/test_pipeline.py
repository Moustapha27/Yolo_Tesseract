import os
import pytest
from unittest import mock
from unittest.mock import patch, MagicMock
import builtins
import cv2
import numpy as np
from PIL import Image
import shutil

from src.kedro_road_sign.pipelines.ocr_pipeline.nodes import (
    configure_tesseract,
    prepare_ocr_data,
    perform_ocr
)

def test_configure_tesseract_sets_env(monkeypatch):
    tesseract_path = shutil.which("tesseract")
    if not tesseract_path:
        pytest.skip("Tesseract n'est pas installé ou introuvable dans le PATH")

    config = {"tessdata_prefix": "/fake/path"}
    monkeypatch.delenv("TESSDATA_PREFIX", raising=False)

    try:
        configure_tesseract(config)
    except RuntimeError as e:
        pytest.fail(f"Erreur inattendue lors de la configuration de Tesseract : {e}")
    
    assert os.environ.get("TESSDATA_PREFIX") == "/fake/path"

@patch("src.kedro_road_sign.pipelines.ocr_pipeline.nodes.pytesseract.get_tesseract_version")
def test_configure_tesseract_raises_if_not_installed(mock_get_version):
    mock_get_version.side_effect = Exception("Not installed")
    with pytest.raises(RuntimeError, match="Tesseract non installé"):
        configure_tesseract({})

@patch("cv2.imread")
@patch("cv2.imwrite")
def test_prepare_ocr_data_creates_files(mock_imwrite, mock_imread, tmp_path):
    # Simule une image numpy pour cv2.imread
    dummy_image = np.ones((100, 100, 3), dtype=np.uint8)
    mock_imread.return_value = dummy_image
    
    yolo_predictions = {
        "image1.jpg": [
            (10, 20, 50, 60, 0.9, 1),
            (5, 5, 25, 25, 0.8, 0)
        ]
    }
    output_dir = tmp_path / "output"
    output_dir = str(output_dir)
    
    result = prepare_ocr_data(yolo_predictions, output_dir)
    
    # Vérifie que cv2.imwrite a été appelé deux fois (pour les deux régions)
    assert mock_imwrite.call_count == 2
    
    # Vérifie que les fichiers "enregistrés" ont des chemins corrects
    for call in mock_imwrite.call_args_list:
        args, kwargs = call
        path = args[0]  # <-- Le premier argument de imwrite est le chemin du fichier
        # On vérifie que le dossier parent du fichier est bien output_dir ou existe
        dir_path = os.path.dirname(path)
        assert dir_path == output_dir or os.path.exists(dir_path)

@patch("src.kedro_road_sign.pipelines.ocr_pipeline.nodes.pytesseract.image_to_string")
@patch("PIL.Image.open")
def test_perform_ocr_success_and_failure(mock_image_open, mock_ocr):
    # Simule un texte OCR reconnu
    mock_ocr.return_value = "Test Text"
    mock_image_open.return_value = MagicMock()
    
    detected_data = {
        "detected_signs": [
            {
                "roi_path": "fake_path_1.png",
                "original_image": "img.jpg",
                "coordinates": (0,0,10,10),
                "confidence": 0.95
            },
            {
                "roi_path": "fake_path_2.png",
                "original_image": "img2.jpg",
                "coordinates": (5,5,15,15),
                "confidence": 0.85
            }
        ]
    }
    
    # Test OCR succès
    results = perform_ocr(detected_data, lang="fra")
    assert "ocr_results" in results
    assert len(results["ocr_results"]) == 2
    for res in results["ocr_results"]:
        assert res["ocr_success"] is True
        assert res["detected_text"] == "Test Text"
        assert "error" not in res
    
    # Test OCR avec exception
    def raise_exception(*args, **kwargs):
        raise RuntimeError("OCR failed")
    
    mock_ocr.side_effect = raise_exception
    
    results_error = perform_ocr(detected_data, lang="fra")
    for res in results_error["ocr_results"]:
        assert res["ocr_success"] is False
        assert res["detected_text"] == ""
        assert "error" in res
