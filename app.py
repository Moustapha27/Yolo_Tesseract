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
import time

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

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vision AI - Analyse de Signalisation</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <style>
        :root {
            --primary: #4361ee;
            --secondary: #3f37c9;
            --accent: #4895ef;
            --success: #4cc9f0;
            --light: #f8f9fa;
            --dark: #212529;
            --danger: #f72585;
            --warning: #f8961e;
        }
        
        body {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            margin: 0;
            padding: 0;
            color: var(--dark);
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        header {
            text-align: center;
            margin-bottom: 2rem;
            padding: 1rem;
            background: rgba(255, 255, 255, 0.8);
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        }
        
        h1 {
            color: var(--primary);
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        
        .subtitle {
            color: var(--secondary);
            font-size: 1.1rem;
            margin-bottom: 1.5rem;
        }
        
        .upload-section {
            background: white;
            border-radius: 15px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .upload-section:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        }
        
        .upload-area {
            border: 3px dashed var(--accent);
            border-radius: 10px;
            padding: 2rem;
            margin: 1rem 0;
            cursor: pointer;
            transition: all 0.3s ease;
            background-color: rgba(67, 97, 238, 0.05);
        }
        
        .upload-area:hover {
            background-color: rgba(67, 97, 238, 0.1);
        }
        
        .upload-icon {
            font-size: 3rem;
            color: var(--accent);
            margin-bottom: 1rem;
        }
        
        .file-input {
            display: none;
        }
        
        .btn {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            padding: 0.8rem 1.8rem;
            border: none;
            border-radius: 50px;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 500;
            margin: 0.5rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(67, 97, 238, 0.3);
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(67, 97, 238, 0.4);
        }
        
        .btn:active {
            transform: translateY(0);
        }
        
        .btn-outline {
            background: transparent;
            border: 2px solid var(--primary);
            color: var(--primary);
            box-shadow: none;
        }
        
        .btn-outline:hover {
            background: rgba(67, 97, 238, 0.1);
        }
        
        .results-container {
            display: flex;
            flex-wrap: wrap;
            gap: 2rem;
            margin-top: 2rem;
        }
        
        .result-card {
            flex: 1;
            min-width: 300px;
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        }
        
        .result-header {
            display: flex;
            align-items: center;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--light);
        }
        
        .result-icon {
            font-size: 1.5rem;
            margin-right: 0.8rem;
            color: var(--accent);
        }
        
        .result-title {
            font-size: 1.3rem;
            font-weight: 600;
            color: var(--primary);
            margin: 0;
        }
        
        .result-image {
            width: 100%;
            border-radius: 10px;
            margin-bottom: 1rem;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }
        
        .result-image:hover {
            transform: scale(1.02);
        }
        
        .detections-list {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        
        .detection-item {
            background: var(--light);
            border-radius: 8px;
            padding: 0.8rem;
            margin-bottom: 0.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .detection-class {
            font-weight: 500;
            color: var(--secondary);
        }
        
        .detection-confidence {
            background: var(--accent);
            color: white;
            padding: 0.2rem 0.5rem;
            border-radius: 20px;
            font-size: 0.8rem;
        }
        
        .ocr-text {
            background: var(--light);
            border-radius: 10px;
            padding: 1rem;
            min-height: 200px;
            white-space: pre-wrap;
            font-family: 'Courier New', monospace;
            border-left: 4px solid var(--accent);
        }
        
        .stats-card {
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }
        
        .stat-item {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.5rem;
        }
        
        .stat-label {
            font-weight: 500;
            color: var(--secondary);
        }
        
        .stat-value {
            font-weight: 600;
        }
        
        .loading {
            text-align: center;
            padding: 2rem;
            display: none;
        }
        
        .spinner {
            border: 5px solid rgba(67, 97, 238, 0.1);
            border-top: 5px solid var(--primary);
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 1rem;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error {
            background: rgba(247, 37, 133, 0.1);
            border-left: 4px solid var(--danger);
            color: var(--danger);
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            display: none;
        }
        
        .success {
            background: rgba(76, 201, 240, 0.1);
            border-left: 4px solid var(--success);
            color: var(--success);
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            display: none;
        }
        
        .file-info {
            margin-top: 1rem;
            font-size: 0.9rem;
            color: var(--secondary);
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 1rem;
            }
            
            .results-container {
                flex-direction: column;
            }
            
            h1 {
                font-size: 2rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1><i class="fas fa-traffic-light"></i> Vision AI - Analyse de Signalisation</h1>
            <p class="subtitle">Détection de panneaux routiers et reconnaissance de texte en temps réel</p>
        </header>
        
        <div class="upload-section">
            <h2><i class="fas fa-cloud-upload-alt"></i> Analyse d'image</h2>
            <p>Chargez une image pour détecter les panneaux de signalisation et extraire le texte</p>
            
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="upload-area" id="uploadArea">
                    <div class="upload-icon">
                        <i class="fas fa-images"></i>
                    </div>
                    <p>Glissez-déposez votre image ici ou cliquez pour sélectionner</p>
                    <input type="file" id="imageFile" name="file" accept="image/*" required class="file-input">
                    <button type="button" class="btn btn-outline" id="selectFileBtn">Sélectionner un fichier</button>
                </div>
                <div id="fileInfo" class="file-info"></div>
                <button type="submit" class="btn"><i class="fas fa-search"></i> Analyser l'image</button>
            </form>
        </div>
        
        <div id="loading" class="loading">
            <div class="spinner"></div>
            <h3>Analyse en cours...</h3>
            <p>Détection des panneaux et extraction du texte</p>
        </div>
        
        <div id="error" class="error"></div>
        <div id="success" class="success"></div>
        
        <div id="results" style="display: none;">
            <div class="stats-card">
                <h3><i class="fas fa-chart-bar"></i> Statistiques de l'analyse</h3>
                <div class="stat-item">
                    <span class="stat-label">Panneaux détectés:</span>
                    <span class="stat-value" id="statDetections">0</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Confiance moyenne:</span>
                    <span class="stat-value" id="statConfidence">0%</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Texte extrait:</span>
                    <span class="stat-value" id="statTextFound">Non</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Temps de traitement:</span>
                    <span class="stat-value" id="statProcessingTime">-</span>
                </div>
            </div>
            
            <div class="results-container">
                <div class="result-card">
                    <div class="result-header">
                        <div class="result-icon">
                            <i class="fas fa-search-location"></i>
                        </div>
                        <h3 class="result-title">Détections YOLO</h3>
                    </div>
                    <img id="annotatedImage" class="result-image" alt="Image annotée">
                    <h4><i class="fas fa-list"></i> Détails des détections</h4>
                    <ul class="detections-list" id="detectionsList"></ul>
                </div>
                
                <div class="result-card">
                    <div class="result-header">
                        <div class="result-icon">
                            <i class="fas fa-font"></i>
                        </div>
                        <h3 class="result-title">Texte extrait (OCR)</h3>
                    </div>
                    <div id="ocrText" class="ocr-text">Aucun texte détecté</div>
                    <h4><i class="fas fa-info-circle"></i> Interprétation</h4>
                    <div id="textInterpretation" class="ocr-text">Le texte extrait sera analysé ici pour identifier les panneaux de signalisation.</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Gestion de l'interface d'upload
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('imageFile');
        const selectFileBtn = document.getElementById('selectFileBtn');
        const fileInfo = document.getElementById('fileInfo');
        
        // Cliquer sur la zone ou le bouton ouvre le sélecteur de fichiers
        uploadArea.addEventListener('click', () => fileInput.click());
        selectFileBtn.addEventListener('click', () => fileInput.click());
        
        // Affichage des informations du fichier sélectionné
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                const file = e.target.files[0];
                fileInfo.innerHTML = `
                    <i class="fas fa-file-image"></i> ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)
                `;
            }
        });
        
        // Gestion du glisser-déposer
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.backgroundColor = 'rgba(67, 97, 238, 0.2)';
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.style.backgroundColor = 'rgba(67, 97, 238, 0.05)';
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.backgroundColor = 'rgba(67, 97, 238, 0.05)';
            
            if (e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                const file = e.dataTransfer.files[0];
                fileInfo.innerHTML = `
                    <i class="fas fa-file-image"></i> ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)
                `;
            }
        });
        
        // Soumission du formulaire
        document.getElementById('uploadForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
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

            // Afficher le loading et masquer les résultats
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            document.getElementById('error').style.display = 'none';
            document.getElementById('success').style.display = 'none';
            
            // Démarrer le chrono
            const startTime = performance.now();

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();
                const endTime = performance.now();
                const processingTime = ((endTime - startTime) / 1000).toFixed(2);

                if (response.ok) {
                    // Afficher les résultats
                    document.getElementById('annotatedImage').src = result.annotated_image;
                    document.getElementById('ocrText').textContent = result.ocr_text || 'Aucun texte détecté';
                    
                    // Mettre à jour les statistiques
                    document.getElementById('statDetections').textContent = result.num_detections;
                    document.getElementById('statProcessingTime').textContent = `${processingTime} sec`;
                    
                    // Calculer la confiance moyenne
                    if (result.detections && result.detections.length > 0) {
                        const avgConfidence = result.detections.reduce((sum, det) => sum + det.confidence, 0) / result.detections.length;
                        document.getElementById('statConfidence').textContent = `${(avgConfidence * 100).toFixed(1)}%`;
                    } else {
                        document.getElementById('statConfidence').textContent = '0%';
                    }
                    
                    // Texte trouvé ou non
                    document.getElementById('statTextFound').textContent = 
                        result.ocr_text && result.ocr_text !== 'Aucun texte détecté' ? 'Oui' : 'Non';
                    
                    // Afficher la liste des détections
                    const detectionsList = document.getElementById('detectionsList');
                    detectionsList.innerHTML = '';
                    
                    if (result.detections && result.detections.length > 0) {
                        result.detections.forEach(det => {
                            const li = document.createElement('li');
                            li.className = 'detection-item';
                            li.innerHTML = `
                                <span class="detection-class">${det.class}</span>
                                <span class="detection-confidence">${(det.confidence * 100).toFixed(1)}%</span>
                            `;
                            detectionsList.appendChild(li);
                        });
                    } else {
                        detectionsList.innerHTML = '<li class="detection-item">Aucun panneau détecté</li>';
                    }
                    
                    // Interprétation du texte
                    const textInterpretation = document.getElementById('textInterpretation');
                    if (result.ocr_text && result.ocr_text !== 'Aucun texte détecté') {
                        const knownSigns = ['STOP', 'YIELD', 'CEDER', 'CEDEZ', 'ARRET', 'ATTENTION'];
                        const foundSign = knownSigns.find(sign => 
                            result.ocr_text.toUpperCase().includes(sign)
                        );
                        
                        if (foundSign) {
                            textInterpretation.innerHTML = `
                                <strong>Panneau identifié:</strong> ${foundSign}<br><br>
                                <strong>Interprétation:</strong> ${getSignInterpretation(foundSign)}
                            `;
                        } else {
                            textInterpretation.textContent = 
                                "Le texte extrait ne correspond à aucun panneau de signalisation connu.";
                        }
                    } else {
                        textInterpretation.textContent = "Aucun texte significatif détecté.";
                    }
                    
                    // Afficher le tout
                    document.getElementById('results').style.display = 'block';
                    document.getElementById('success').style.display = 'block';
                    document.getElementById('success').textContent = 
                        `Analyse terminée avec succès en ${processingTime} secondes.`;
                    
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
            errorDiv.innerHTML = `<i class="fas fa-exclamation-circle"></i> ${message}`;
            errorDiv.style.display = 'block';
            document.getElementById('results').style.display = 'none';
            document.getElementById('loading').style.display = 'none';
        }
        
        function getSignInterpretation(sign) {
            const interpretations = {
                'STOP': 'Arrêt obligatoire. Vous devez marquer un arrêt complet.',
                'YIELD': 'Cédez le passage. Ralentissez et cédez le passage si nécessaire.',
                'CEDER': 'Cédez le passage. Même signification que YIELD en français.',
                'CEDEZ': 'Cédez le passage. Variante orthographique de CEDER.',
                'ARRET': 'Arrêt obligatoire. Equivalent français de STOP.',
                'ATTENTION': 'Zone de danger ou nécessitant une attention particulière.'
            };
            
            return interpretations[sign] || 'Panneau de signalisation standard.';
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
            start_time = time.time()
            
            # Traitement Roboflow YOLO
            annotated_image, detections = process_image_with_roboflow(temp_path)
            
            # Conversion de l'image annotée en base64
            annotated_image_b64 = image_to_base64(annotated_image)
            
            # Extraction OCR
            ocr_text = extract_text_with_ocr(temp_path)
            
            processing_time = time.time() - start_time
            
            response_data = {
                'success': True,
                'annotated_image': annotated_image_b64,
                'ocr_text': ocr_text,
                'detections': detections,
                'num_detections': len(detections),
                'processing_time': round(processing_time, 2)
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
        'api_key_configured': 'Oui' if ROBOFLOW_API_KEY != "YOUR_API_KEY_HERE" else 'Non',
        'tesseract': 'Disponible' if pytesseract.get_tesseract_version() else 'Indisponible'
    }
    
    return jsonify(status)

if __name__ == '__main__':
    print("🚀 Démarrage de l'API Flask Roboflow + OCR")
    print(f"📁 Dossier uploads: {UPLOAD_FOLDER}")
    print(f"🤖 Modèle Roboflow: {ROBOFLOW_MODEL_ID}")
    print(f"🔑 API Key configurée: {'Oui' if ROBOFLOW_API_KEY != 'YOUR_API_KEY_HERE' else 'Non - À configurer!'}")
    print("🌐 Interface disponible sur: http://localhost:8000")
    print("💊 Health check: http://localhost:8000/health")
    
    app.run(debug=True, host='0.0.0.0', port=8000)