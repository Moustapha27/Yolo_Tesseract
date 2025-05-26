"""
Pipeline YOLOv8-10 pour la détection de panneaux routiers
=====================================================
Ce module définit tous les nœuds et la pipeline nécessaires pour entraîner 
et évaluer un modèle YOLOv8-10 pour la détection de panneaux routiers.
"""

import os
import shutil
import yaml
import numpy as np
from PIL import Image
from typing import Dict, List, Any
from pathlib import Path
from ultralytics import YOLO

from ultralytics import YOLO

def train_yolov8_model(data_yaml: dict) -> YOLO:
    """
    Entraîne un modèle YOLOv8 à partir d'un fichier data.yml.
    """
    # On suppose que tu veux un modèle YOLOv8n (le plus petit)
    model = YOLO("yolov8n.pt")  # ou "yolov8s.pt" pour une version plus grosse

    # Dossier contenant le fichier data.yml
    data_path = data_yaml.get("__path__", "data/01_raw/data.yaml")

    model.train(
        data=data_path,
        epochs=20,
        imgsz=640,
        batch=16,
        patience=5
    )

    return model
