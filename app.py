from flask import Flask, request, jsonify
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from io import BytesIO
import os

app = Flask(__name__)

# Laden Sie das Modell
model = VisionEncoderDecoderModel.from_pretrained('sk_invoice_receipts')
feature_extractor = ViTImageProcessor.from_pretrained('sk_invoice_receipts')
tokenizer = AutoTokenizer.from_pretrained('sk_invoice_receipts')

device = torch.device('cpu')
model.to(device)

# Ordner zum Speichern der Bilder
IMAGE_SAVE_PATH = "saved_images"
os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)

@app.route('/')
def home():
    return "Welcome to the Image Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    # Get image buffer from request
    image_data = request.data
    image = Image.open(BytesIO(image_data))

    # Bild auf dem Server speichern
    image_filename = os.path.join(IMAGE_SAVE_PATH, 'uploaded_image.png')
    image.save(image_filename)

    # Preprocess the image
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
    
    # Generate predictions
    output_ids = model.generate(pixel_values)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    
    return jsonify({'prediction': preds[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
