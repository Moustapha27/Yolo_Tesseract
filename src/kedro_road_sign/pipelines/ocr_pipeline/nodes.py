import pytesseract
import cv2
import os
from typing import List, Dict
from PIL import Image

def configure_tesseract(tesseract_config: dict) -> None:
    """Configure Tesseract OCR"""
    if tesseract_config.get("tessdata_prefix"):
        os.environ['TESSDATA_PREFIX'] = tesseract_config["tessdata_prefix"]
    try:
        pytesseract.get_tesseract_version()
    except:
        raise RuntimeError("Tesseract non installé")

def prepare_ocr_data(yolo_predictions: dict, output_dir: str) -> dict:
    """Prépare les régions détectées pour l'OCR"""
    os.makedirs(output_dir, exist_ok=True)
    detected_signs = []
    
    for img_path, detections in yolo_predictions.items():
        img = cv2.imread(img_path)
        for i, (x1, y1, x2, y2, conf, cls) in enumerate(detections):
            roi = img[int(y1):int(y2), int(x1):int(x2)]
            roi_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_sign_{i}.png")
            cv2.imwrite(roi_path, roi)
            detected_signs.append({
                "original_image": img_path,
                "roi_path": roi_path,
                "coordinates": (x1, y1, x2, y2),
                "confidence": conf
            })
    return {"detected_signs": detected_signs}

def perform_ocr(detected_data: dict, lang: str = "fra") -> dict:
    """Exécute l'OCR sur les régions détectées"""
    results = []
    for sign in detected_data["detected_signs"]:
        try:
            text = pytesseract.image_to_string(Image.open(sign["roi_path"]), lang=lang).strip()
            results.append({**sign, "detected_text": text, "ocr_success": bool(text)})
        except Exception as e:
            results.append({**sign, "detected_text": "", "ocr_success": False, "error": str(e)})
    return {"ocr_results": results}