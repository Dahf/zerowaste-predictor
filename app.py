from flask import Flask, request, jsonify
from PIL import Image
from transformers import VisionEncoderDecoderModel, DonutProcessor
import torch
from io import BytesIO
import os
import re

app = Flask(__name__)

# Laden Sie das Modell
model = VisionEncoderDecoderModel.from_pretrained('sk_invoice_receipts')
processor = DonutProcessor.from_pretrained('sk_invoice_receipts')

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
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    model.eval()
    with torch.no_grad():
        task_prompt = "<s_receipt>"
        decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
        decoder_input_ids = decoder_input_ids.to(device)
        generated_outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.decoder.config.max_position_embeddings, 
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            early_stopping=True,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True
        )

        decoded_text = processor.batch_decode(generated_outputs.sequences)[0]
        decoded_text = decoded_text.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        decoded_text = re.sub(r"<.*?>", "", decoded_text, count=1).strip()  # remove first task start token
        decoded_text = processor.token2json(decoded_text)
        return decoded_text

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
