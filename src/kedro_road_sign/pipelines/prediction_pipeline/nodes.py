"""
This is a boilerplate pipeline 'prediction_pipeline'
generated using Kedro 0.19.12
"""
from ultralytics import YOLO
import pytesseract
import cv2
from PIL import Image
from typing import Dict, List

def load_models(yolo_model_path: str) -> dict:
    """Charge les modèles nécessaires"""
    return {"yolo_model": YOLO(yolo_model_path)}

def predict_with_yolo(models: dict, image_paths: List[str]) -> dict:
    """Effectue les prédictions YOLO"""
    predictions = {}
    for img_path in image_paths:
        results = models["yolo_model"](img_path)
        predictions[img_path] = [box.data[0].tolist() for box in results[0].boxes]
    return {"yolo_predictions": predictions}

def extract_and_recognize_text(models: dict, yolo_predictions: dict) -> dict:
    """Combine détection YOLO et OCR"""
    combined_results = []
    for img_path, detections in yolo_predictions["yolo_predictions"].items():
        img = cv2.imread(img_path)
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            roi = Image.fromarray(cv2.cvtColor(img[int(y1):int(y2), int(x1):int(x2)], cv2.COLOR_BGR2RGB))
            text = pytesseract.image_to_string(roi, lang="fra").strip()
            combined_results.append({
                "image_path": img_path,
                "coordinates": (x1, y1, x2, y2),
                "confidence": conf,
                "class": cls,
                "detected_text": text
            })
    return {"combined_predictions": combined_results}