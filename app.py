from flask import Flask, request, render_template_string, jsonify, send_file
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import os
from werkzeug.utils import secure_filename
import tempfile

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Configuration Roboflow
ROBOFLOW_API_KEY = "cCj3D5dgbnYlQ5mVtlyV"  # Remplacez par votre clé API
ROBOFLOW_MODEL_ID = "traffic-sign-detection-yolov8-awuus/1"  # Remplacez par votre model_id

# Créer le dossier uploads s'il n'existe pas
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialiser le client Roboflow
try:
    roboflow_client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=ROBOFLOW_API_KEY
    )
    print(f"Client Roboflow initialisé avec le modèle: {ROBOFLOW_MODEL_ID}")
except Exception as e:
    print(f"Erreur lors de l'initialisation du client Roboflow: {e}")
    roboflow_client = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(image_array):
    """Convertit une image numpy en base64 pour l'affichage HTML"""
    _, buffer = cv2.imencode('.jpg', image_array)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"

def process_image_with_roboflow(image_path):
    """Traite l'image avec Roboflow YOLO et retourne l'image annotée"""
    if roboflow_client is None:
        raise Exception("Client Roboflow non disponible")
    
    # Prédiction avec Roboflow
    result = roboflow_client.infer(image_path, model_id=ROBOFLOW_MODEL_ID)
    
    # Charger l'image originale
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # Dessiner les détections
    detections = []
    if 'predictions' in result:
        for prediction in result['predictions']:
            # Extraire les coordonnées et informations
            x = prediction['x']
            y = prediction['y']
            width = prediction['width']
            height = prediction['height']
            confidence = prediction['confidence']
            class_name = prediction['class']
            
            # Calculer les coordonnées du rectangle
            left = x - width / 2
            top = y - height / 2
            right = x + width / 2
            bottom = y + height / 2
            
            # Dessiner le rectangle
            draw.rectangle([left, top, right, bottom], outline='red', width=3)
            
            # Ajouter le label
            label = f"{class_name}: {confidence:.2f}"
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Calculer la taille du texte pour le fond
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Dessiner le fond du texte
            draw.rectangle([left, top - text_height - 4, left + text_width + 4, top], fill='red')
            draw.text((left + 2, top - text_height - 2), label, fill='white', font=font)
            
            # Ajouter à la liste des détections
            detections.append({
                'class': class_name,
                'confidence': round(confidence, 3),
                'bbox': [left, top, right, bottom]
            })
    
    # Convertir PIL Image en numpy array pour compatibilité
    annotated_image = np.array(image)
    
    return annotated_image, detections

def extract_text_with_ocr(image_path):
    """Extrait le texte de l'image avec Tesseract OCR"""
    try:
        # Ouvrir l'image avec PIL
        image = Image.open(image_path)
        
        # Convertir en numpy array pour OpenCV
        img_array = np.array(image)
        
        # Convertir en BGR si nécessaire (OpenCV utilise BGR)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Prétraitement pour améliorer l'OCR
        # 1. Convertir en niveaux de gris
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        
        # 2. Augmenter le contraste
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # 3. Appliquer un flou gaussien léger pour réduire le bruit
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # 4. Binarisation adaptative
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # 5. Morphologie pour nettoyer
        kernel = np.ones((2,2), np.uint8)
        processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
        
        # Convertir back en PIL Image
        processed_image = Image.fromarray(processed)
        
        # Essayer plusieurs configurations OCR
        configs = [
            r'--oem 3 --psm 8 -l eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ',  # Lettres majuscules seulement
            r'--oem 3 --psm 7 -l eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ',  # Ligne de texte, lettres seulement
            r'--oem 3 --psm 6 -l eng',  # Anglais seulement, block de texte
            r'--oem 3 --psm 8 -l eng',  # Anglais seulement, mot unique
            r'--oem 3 --psm 13 -l eng', # Anglais seulement, ligne brute
            r'--oem 3 --psm 6 -l fra+eng',  # Français + anglais
        ]
        
        all_results = []
        
        for config in configs:
            try:
                # Essayer avec l'image prétraitée
                text = pytesseract.image_to_string(processed_image, config=config).strip()
                if text:
                    all_results.append(text)
                
                # Essayer aussi avec l'image originale
                text_orig = pytesseract.image_to_string(image, config=config).strip()
                if text_orig:
                    all_results.append(text_orig)
                    
            except Exception:
                continue
        
        # Si aucun texte trouvé, essayer avec redimensionnement
        if not all_results:
            # Redimensionner l'image (parfois l'OCR marche mieux avec des images plus grandes)
            width, height = image.size
            resized = image.resize((width * 3, height * 3), Image.LANCZOS)
            
            try:
                text = pytesseract.image_to_string(resized, config=r'--oem 3 --psm 8 -l eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ').strip()
                if text:
                    all_results.append(text)
            except:
                pass
        
        # Post-traitement : correction des erreurs communes
        corrected_results = []
        for text in all_results:
            corrected = text
            
            # Corrections communes pour les panneaux de signalisation
            corrections = {
                'Gran)': 'STOP',
                'Gran': 'STOP', 
                'Gr0n': 'STOP',
                'St0p': 'STOP',
                'St0P': 'STOP',
                'STDP': 'STOP',
                'STQP': 'STOP',
                'STOР': 'STOP',  # O cyrillique vers O latin
                'STОР': 'STOP',  # P cyrillique vers P latin
                'YEILD': 'YIELD',
                'YIEI_D': 'YIELD',
                'CEDER': 'CEDER',
                'CEDEZ': 'CEDEZ',
            }
            
            # Appliquer les corrections
            for wrong, right in corrections.items():
                if wrong.upper() in corrected.upper():
                    corrected = corrected.upper().replace(wrong.upper(), right)
                    break
            
            # Nettoyer les caractères indésirables
            corrected = ''.join(char for char in corrected if char.isalnum() or char.isspace())
            
            if corrected:
                corrected_results.append(corrected)
        
        # Retourner le meilleur résultat
        if corrected_results:
            # Privilégier les mots reconnus comme panneaux
            known_signs = ['STOP', 'YIELD', 'CEDER', 'CEDEZ', 'ARRET', 'ATTENTION']
            for result in corrected_results:
                if any(sign in result.upper() for sign in known_signs):
                    return result
            
            # Sinon, retourner le plus long
            return max(corrected_results, key=len)
        
        return "Aucun texte détecté"
        
    except Exception as e:
        return f"Erreur OCR: {str(e)}"

# Template HTML intégré
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>API YOLO + OCR</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .upload-section {
            border: 2px dashed #007bff;
            border-radius: 10px;
            padding: 30px;
            text-align: center;
            margin-bottom: 30px;
            background-color: #f8f9fa;
        }
        .file-input {
            margin: 20px 0;
        }
        input[type="file"] {
            margin: 10px 0;
        }
        .btn {
            background-color: #007bff;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin: 10px;
        }
        .btn:hover {
            background-color: #0056b3;
        }
        .results {
            margin-top: 30px;
        }
        .result-section {
            display: flex;
            gap: 20px;
            margin-top: 20px;
        }
        .image-result {
            flex: 1;
        }
        .text-result {
            flex: 1;
        }
        .result-image {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .ocr-text {
            background-color: #f8f9fa;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            min-height: 200px;
            white-space: pre-wrap;
            font-family: monospace;
        }
        .loading {
            text-align: center;
            color: #007bff;
            font-style: italic;
        }
        .error {
            color: #dc3545;
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
        }
        .success {
            color: #155724;
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
        }
        @media (max-width: 768px) {
            .result-section {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 API YOLO + OCR</h1>
        
        <div class="upload-section">
            <h3>📤 Upload une image</h3>
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="file-input">
                    <input type="file" id="imageFile" name="file" accept="image/*" required>
                </div>
                <button type="submit" class="btn">🚀 Analyser l'image</button>
            </form>
        </div>

        <div id="loading" class="loading" style="display: none;">
            ⏳ Analyse en cours... Détection YOLO + extraction OCR...
        </div>

        <div id="error" class="error" style="display: none;"></div>

        <div id="results" class="results" style="display: none;">
            <h3>📊 Résultats de l'analyse</h3>
            <div class="result-section">
                <div class="image-result">
                    <h4>🎯 Image avec détections YOLO</h4>
                    <img id="annotatedImage" class="result-image" alt="Image annotée">
                </div>
                <div class="text-result">
                    <h4>📝 Texte extrait (OCR)</h4>
                    <div id="ocrText" class="ocr-text"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('uploadForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const fileInput = document.getElementById('imageFile');
            const file = fileInput.files[0];
            
            if (!file) {
                showError('Veuillez sélectionner une image');
                return;
            }

            // Vérifier la taille du fichier (16MB max)
            if (file.size > 16 * 1024 * 1024) {
                showError('Le fichier est trop volumineux (16MB maximum)');
                return;
            }

            const formData = new FormData();
            formData.append('file', file);

            // Afficher le loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            document.getElementById('error').style.display = 'none';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (response.ok) {
                    // Afficher les résultats
                    document.getElementById('annotatedImage').src = result.annotated_image;
                    document.getElementById('ocrText').textContent = result.ocr_text || 'Aucun texte détecté';
                    document.getElementById('results').style.display = 'block';
                } else {
                    showError(result.error || 'Erreur lors du traitement');
                }
            } catch (error) {
                showError('Erreur de connexion: ' + error.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        });

        function showError(message) {
            const errorDiv = document.getElementById('error');
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
            document.getElementById('results').style.display = 'none';
            document.getElementById('loading').style.display = 'none';
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Page d'accueil avec interface d'upload"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint pour traiter l'image avec YOLO + OCR"""
    try:
        # Vérifier qu'un fichier a été envoyé
        if 'file' not in request.files:
            return jsonify({'error': 'Aucun fichier envoyé'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Aucun fichier sélectionné'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Type de fichier non autorisé'}), 400
        
        # Sauvegarder le fichier temporairement
        filename = secure_filename(file.filename)
        temp_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(temp_path)
        
        try:
            # Traitement Roboflow YOLO
            annotated_image, detections = process_image_with_roboflow(temp_path)
            
            # Conversion de l'image annotée en base64
            annotated_image_b64 = image_to_base64(annotated_image)
            
            # Extraction OCR
            ocr_text = extract_text_with_ocr(temp_path)
            
            response_data = {
                'success': True,
                'annotated_image': annotated_image_b64,
                'ocr_text': ocr_text,
                'detections': detections,
                'num_detections': len(detections)
            }
            
            return jsonify(response_data)
            
        finally:
            # Nettoyer le fichier temporaire
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        return jsonify({'error': f'Erreur lors du traitement: {str(e)}'}), 500

@app.route('/health')
def health():
    """Endpoint de santé pour vérifier le statut de l'API"""
    status = {
        'status': 'OK',
        'roboflow_client': 'Disponible' if roboflow_client is not None else 'Indisponible',
        'model_id': ROBOFLOW_MODEL_ID,
        'api_key_configured': 'Oui' if ROBOFLOW_API_KEY != "YOUR_API_KEY_HERE" else 'Non'
    }
    
    # Test OCR
    try:
        pytesseract.get_tesseract_version()
        status['tesseract'] = 'Disponible'
    except:
        status['tesseract'] = 'Indisponible'
    
    return jsonify(status)

if __name__ == '__main__':
    print("🚀 Démarrage de l'API Flask Roboflow + OCR")
    print(f"📁 Dossier uploads: {UPLOAD_FOLDER}")
    print(f"🤖 Modèle Roboflow: {ROBOFLOW_MODEL_ID}")
    print(f"🔑 API Key configurée: {'Oui' if ROBOFLOW_API_KEY != 'YOUR_API_KEY_HERE' else 'Non - À configurer!'}")
    print("🌐 Interface disponible sur: http://localhost:8000")
    print("💊 Health check: http://localhost:8000/health")
    
    app.run(debug=True, host='0.0.0.0', port=8000)