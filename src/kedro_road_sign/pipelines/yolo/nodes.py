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
import pickle

def train_yolov8_model(data_yaml: dict) -> YOLO:
    """
    Entraîne un modèle YOLOv8 
    """
    model = YOLO("yolov8n.pt")  

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


def save_yolov8_model(model: YOLO, filepath: str) -> None:
    """
    Sauvegarde le modèle YOLOv8 dans un fichier pickle.
    """
    with open(filepath, "wb") as f:
        pickle.dump(model, f)

def evaluate_yolov8_model(model: YOLO, data_yaml: dict) -> Dict[str, Any]:
    """
    Évalue le modèle YOLOv8 sur les données de validation et retourne les métriques.
    """
    data_path = data_yaml.get("__path__", "data/01_raw/data.yaml")
    
    metrics = model.val(data=data_path)
    
    results = metrics.results_dict

    print("\n📊 Résultats de l'évaluation YOLOv8 :")
    for metric, value in results.items():
        print(f"- {metric}: {value}")

    return {
        "metrics": results
    }
