from flask import Flask, request, send_file
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForTokenClassification, AutoProcessor
from datasets import load_dataset
import torch
from io import BytesIO
import os
import numpy as np

app = Flask(__name__)

# Laden des Modells und Prozessors
processor = AutoProcessor.from_pretrained("sk_invoice_receipts", apply_ocr=True)
model = AutoModelForTokenClassification.from_pretrained("sk_invoice_receipts")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Laden des Datensatzes und definieren von ID-zu-Label und Label-zu-Farbe Mapping
dataset = load_dataset("Theivaprakasham/wildreceipt", split="test")
labels = dataset.features['ner_tags'].feature.names
id2label = {v: k for v, k in enumerate(labels)}
label2color = {
    "Date_key": 'red',
    "Date_value": 'green',
    "Ignore": 'orange',
    "Others": 'orange',
    "Prod_item_key": 'red',
    "Prod_item_value": 'green',
    "Prod_price_key": 'red',
    "Prod_price_value": 'green',
    "Prod_quantity_key": 'red',
    "Prod_quantity_value": 'green',
    "Store_addr_key": 'red',
    "Store_addr_value": 'green',
    "Store_name_key": 'red',
    "Store_name_value": 'green',
    "Subtotal_key": 'red',
    "Subtotal_value": 'green',
    "Tax_key": 'red',
    "Tax_value": 'green',
    "Tel_key": 'red',
    "Tel_value": 'green',
    "Time_key": 'red',
    "Time_value": 'green',
    "Tips_key": 'red',
    "Tips_value": 'green',
    "Total_key": 'red',
    "Total_value": 'blue'
}

# Ordner zum Speichern der Bilder
IMAGE_SAVE_PATH = "saved_images"
os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)

def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]

def iob_to_label(label):
    return label

def process_image(image):
    width, height = image.size

    # encode
    encoding = processor(image, truncation=True, return_offsets_mapping=True, return_tensors="pt")
    offset_mapping = encoding.pop('offset_mapping')

    # forward pass
    encoding = {k: v.to(device) for k, v in encoding.items()}
    outputs = model(**encoding)

    # get predictions
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    token_boxes = encoding.bbox.squeeze().tolist()

    # only keep non-subword predictions
    is_subword = np.array(offset_mapping.squeeze().tolist())[:, 0] != 0
    true_predictions = [id2label[pred] for idx, pred in enumerate(predictions) if not is_subword[idx]]
    true_boxes = [unnormalize_box(box, width, height) for idx, box in enumerate(token_boxes) if not is_subword[idx]]

    # draw predictions over the image
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for prediction, box in zip(true_predictions, true_boxes):
        predicted_label = iob_to_label(prediction)
        draw.rectangle(box, outline=label2color[predicted_label])
        draw.text((box[0] + 10, box[1] - 10), text=predicted_label, fill=label2color[predicted_label], font=font)
    
    return image

@app.route('/')
def home():
    return "Welcome to the Image Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    # Bilddaten aus dem Request erhalten
    image_data = request.data
    image = Image.open(BytesIO(image_data))

    # Bildverarbeitung und Vorhersage
    processed_image = process_image(image)

    # Bild im Ordner speichern
    image_filename = os.path.join(IMAGE_SAVE_PATH, 'annotated_image.png')
    processed_image.save(image_filename)

    # Bild in einen Puffer speichern und zurückgeben
    buf = BytesIO()
    processed_image.save(buf, format="PNG")
    buf.seek(0)

    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
