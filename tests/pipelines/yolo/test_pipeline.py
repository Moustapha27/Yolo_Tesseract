import pytest
import tempfile
import os
import pickle
from unittest.mock import patch, MagicMock

from src.kedro_road_sign.pipelines.yolo.nodes import train_yolov8_model, save_yolov8_model, evaluate_yolov8_model

@pytest.fixture
def dummy_data_yaml():
    return {"__path__": "dummy/data.yaml"}

@patch("src.kedro_road_sign.pipelines.yolo.nodes.YOLO")
def test_train_yolov8_model(mock_yolo, dummy_data_yaml):
    # Mock le modèle et la méthode train
    mock_model = MagicMock()
    mock_yolo.return_value = mock_model

    model = train_yolov8_model(dummy_data_yaml)

    # Assert que YOLO est bien instancié avec le bon modèle de base
    mock_yolo.assert_called_once_with("yolov8n.pt")

    # Assert que la méthode train est appelée
    mock_model.train.assert_called_once_with(
        data="dummy/data.yaml",
        epochs=20,
        imgsz=640,
        batch=16,
        patience=5
    )

    assert model == mock_model

def test_save_yolov8_model():
    # Crée un modèle fictif (n'importe quoi ici)
    dummy_model = {"mock": "model"}
    
    # Sauvegarde dans un fichier temporaire
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        filepath = tmp.name
    
    try:
        save_yolov8_model(dummy_model, filepath)
        with open(filepath, "rb") as f:
            loaded_model = pickle.load(f)
        assert loaded_model == dummy_model
    finally:
        os.remove(filepath)

@patch("src.kedro_road_sign.pipelines.yolo.nodes.YOLO")
def test_evaluate_yolov8_model(mock_yolo, dummy_data_yaml):
    # Crée un mock de modèle
    mock_model = MagicMock()
    
    # Simule le retour de .val()
    mock_val = MagicMock()
    mock_val.results_dict = {
        "precision": 0.85,
        "recall": 0.80
    }
    mock_model.val.return_value = mock_val

    result = evaluate_yolov8_model(mock_model, dummy_data_yaml)

    mock_model.val.assert_called_once_with(data="dummy/data.yaml")
    assert "metrics" in result
    assert result["metrics"]["precision"] == 0.85
