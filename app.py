from flask import Flask, request, jsonify
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from io import BytesIO

app = Flask(__name__)

# Laden Sie das Modell
model = VisionEncoderDecoderModel.from_pretrained('sk_invoice_receipts')
feature_extractor = ViTImageProcessor.from_pretrained(
    'sk_invoice_receipts')
tokenizer = AutoTokenizer.from_pretrained('sk_invoice_receipts')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {'max_length': max_length, 'num_beams': num_beams}

@app.route('/')
def home():
    return "Welcome to the Image Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    # Get image buffer from request
    image_data = request.data
    image = Image.open(BytesIO(image_data))

    # Get image buffer from request
    pixel_values = feature_extractor(
        images=[image], return_tensors='pt').pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    
    return preds[0]

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
