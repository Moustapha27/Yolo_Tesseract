import os
import shutil
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.kedro_road_sign.pipelines.yolo.nodes import (
    validate_yolo_structure,
    train_yolov8,
    evaluate_yolov8
)

@pytest.fixture
def fake_yolo_dataset(tmp_path):
    """Crée une structure de dossier simulant un dataset YOLO valide"""
    data_path = tmp_path / "yolo_data"
    data_path.mkdir()

    for split in ["train", "test"]:
        split_dir = data_path / split
        (split_dir / "images").mkdir(parents=True)
        (split_dir / "labels").mkdir(parents=True)

    # Création d'un fichier data.yaml
    yaml_file = data_path / "data.yaml"
    yaml_file.write_text("""
train: train/images
val: test/images
nc: 1
names: ['sign']
""")
    return data_path

def test_validate_yolo_structure_success(fake_yolo_dataset):
    result = validate_yolo_structure(str(fake_yolo_dataset))
    assert "yolo_data_path" in result
    assert "data_yaml" in result
    assert os.path.exists(result["data_yaml"])

def test_validate_yolo_structure_missing_folder(tmp_path):
    # Crée uniquement le fichier yaml, pas les sous-dossiers
    data_path = tmp_path / "bad_yolo"
    data_path.mkdir()
    (data_path / "data.yaml").write_text("")

    with pytest.raises(FileNotFoundError):
        validate_yolo_structure(str(data_path))

def test_validate_yolo_structure_missing_yaml(tmp_path):
    # Crée les dossiers mais pas le yaml
    data_path = tmp_path / "no_yaml"
    data_path.mkdir()
    for split in ["train", "test"]:
        (data_path / split / "images").mkdir(parents=True)
        (data_path / split / "labels").mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        validate_yolo_structure(str(data_path))

@patch("src.kedro_road_sign.pipelines.yolo.nodes.YOLO")
def test_train_yolov8(mock_yolo, fake_yolo_dataset, tmp_path):
    # Mock du modèle YOLO
    mock_model = MagicMock()
    mock_yolo.return_value = mock_model
    mock_model.train.return_value = {"some": "result"}

    model_output_dir = tmp_path / "output"
    model_output_dir.mkdir()

    # Simule les fichiers de poids sauvegardés
    weights_dir = model_output_dir / "yolov8_sign_detection" / "weights"
    weights_dir.mkdir(parents=True)
    (weights_dir / "best.pt").write_text("fake weights")
    (weights_dir / "last.pt").write_text("fake weights")

    result = train_yolov8(
        data_info={"data_yaml": str(fake_yolo_dataset / "data.yaml")},
        model_output_dir=str(model_output_dir),
        epochs=1,
        imgsz=32
    )

    assert "best_model" in result
    assert os.path.exists(result["best_model"])
    assert "last_model" in result
    assert os.path.exists(result["last_model"])
    assert "train_results" in result

@patch("src.kedro_road_sign.pipelines.yolo.nodes.YOLO")
def test_evaluate_yolov8(mock_yolo, fake_yolo_dataset):
    # Mock du modèle
    mock_model = MagicMock()
    mock_yolo.return_value = mock_model

    # Simule les métriques
    metrics_mock = MagicMock()
    metrics_mock.box.map50 = 0.75
    metrics_mock.box.map = 0.60
    metrics_mock.box.p = 0.80
    metrics_mock.box.r = 0.70
    mock_model.val.return_value = metrics_mock

    result = evaluate_yolov8(
        model_path="fake_model.pt",
        data_info={"data_yaml": str(fake_yolo_dataset / "data.yaml")}
    )

    assert result["mAP50"] == 0.75
    assert result["mAP50-95"] == 0.60
    assert result["precision"] == 0.80
    assert result["recall"] == 0.70
