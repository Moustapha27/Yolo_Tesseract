from ultralytics import YOLO
import yaml
import os
from typing import Dict

def validate_yolo_structure(data_path: str) -> dict:
    """Valide la structure YOLO existante"""
    required_folders = ['train', 'test']
    required_files = ['data.yaml']
    
    for folder in required_folders:
        folder_path = os.path.join(data_path, folder)
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Dossier manquant: {folder_path}")
        
        images_dir = os.path.join(folder_path, 'images')
        labels_dir = os.path.join(folder_path, 'labels')
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            raise FileNotFoundError(f"Structure YOLO incomplète dans {folder_path}")
    
    data_yaml_path = os.path.join(data_path, 'data.yaml')
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"Fichier data.yaml manquant dans {data_path}")
    
    return {"yolo_data_path": data_path, "data_yaml": data_yaml_path}

def train_yolov8(data_info: dict, model_output_dir: str, epochs: int = 50, imgsz: int = 640) -> dict:
    """Entraîne et sauvegarde le modèle YOLOv8"""
    model = YOLO("yolov8s.pt")
    results = model.train(
        data=data_info["data_yaml"],
        epochs=epochs,
        imgsz=imgsz,
        project=model_output_dir,
        name="yolov8_sign_detection",
        save=True,
        save_period=5
    )
    
    best_model_path = os.path.join(model_output_dir, "yolov8_sign_detection", "weights", "best.pt")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Modèle non sauvegardé à {best_model_path}")
    
    return {
        "best_model": best_model_path,
        "last_model": os.path.join(model_output_dir, "yolov8_sign_detection", "weights", "last.pt"),
        "train_results": results
    }

def evaluate_yolov8(model_path: str, data_info: dict) -> dict:
    """Évalue le modèle YOLOv8"""
    model = YOLO(model_path)
    metrics = model.val(data=data_info["data_yaml"], split="test")
    return {
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
        "precision": metrics.box.p,
        "recall": metrics.box.r
    }