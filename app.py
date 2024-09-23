from flask import Flask, request, jsonify, render_template, send_file
from PIL import Image, ExifTags
from io import BytesIO
import boto3
import json
import os
import re
import base64

app = Flask(__name__)

# SageMaker Endpunktname
ENDPOINT_NAME = 'huggingface-pytorch-inference-2024-09-22-15-58-30-837'  # Füge hier deinen SageMaker-Endpunktnamen ein

# Boto3-Client für SageMaker mit der Region 'eu-central-1'
sagemaker_client = boto3.client('sagemaker-runtime', region_name='eu-central-1')

# Ordner zum Speichern der Bilder
IMAGE_SAVE_PATH = "saved_images"
os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)

def correct_image_orientation(image):
    """Automatische Korrektur der Bildausrichtung basierend auf Exif-Daten."""
    try:
        # Exif-Daten laden
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        
        # Wenn Exif-Daten vorhanden sind, Bild ausrichten
        exif = image._getexif()
        if exif is not None:
            orientation_value = exif.get(orientation)
            if orientation_value == 3:
                image = image.rotate(180, expand=True)
            elif orientation_value == 6:
                image = image.rotate(270, expand=True)
            elif orientation_value == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # Falls keine Exif-Daten vorhanden sind, nichts tun
        pass
    return image

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Get the image from the form data
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file", 400
    
    # Open the image and correct orientation
    image = Image.open(file)
    image = correct_image_orientation(image)
    
    # Save the corrected image temporarily to return it
    image_io = BytesIO()
    image.save(image_io, format='PNG')
    image_io.seek(0)

    # Serve the image back to the frontend
    return send_file(image_io, mimetype='image/png')

@app.route('/predict', methods=['POST'])
def predict():
    # Get image buffer from request
    image_data = request.data
    image = Image.open(BytesIO(image_data))

    # Bildausrichtung korrigieren
    image = correct_image_orientation(image)

    # Konvertiere das Bild in Base64
    image_bytes = BytesIO()
    image.save(image_bytes, format='PNG')
    image_base64 = base64.b64encode(image_bytes.getvalue()).decode('utf-8')

    # Erstelle JSON mit Base64-kodiertem Bild
    payload = json.dumps({"inputs": image_base64})

    # Senden der Anfrage an den SageMaker-Endpunkt
    response = sagemaker_client.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType='application/json',  # JSON-Daten
        Body=payload
    )

    # Verarbeiten der Antwort von SageMaker
    result = json.loads(response['Body'].read().decode())
    
    # Optional: Falls es nötig ist, entferne spezielle Token aus dem Ergebnis
    decoded_text = re.sub(r"<.*?>", "", result, count=1).strip()

    return jsonify(decoded_text)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5123, debug=True)