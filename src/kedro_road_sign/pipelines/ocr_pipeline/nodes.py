import pytesseract
import cv2
import os

def run_ocr_on_detection(images_folder: str) -> dict:
    results = {}
    for image_name in os.listdir(images_folder):
        if image_name.endswith(".jpg") or image_name.endswith(".png"):
            path = os.path.join(images_folder, image_name)
            image = cv2.imread(path)
            text = pytesseract.image_to_string(image, lang='eng')
            results[image_name] = text
    return results
