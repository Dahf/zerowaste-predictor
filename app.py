from flask import Flask, request, jsonify
from PIL import Image
from transformers import VisionEncoderDecoderModel, DonutProcessor
import torch
from io import BytesIO
import os
import re
import cv2
import numpy as np
import imutils

app = Flask(__name__)

# Laden Sie das Modell
model = VisionEncoderDecoderModel.from_pretrained('sk_invoice_receipts')
processor = DonutProcessor.from_pretrained('sk_invoice_receipts')

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Ordner zum Speichern der Bilder
IMAGE_SAVE_PATH = "saved_images"
os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)
def orient_vertical(img):
    width = img.shape[1]
    height = img.shape[0]
    if width > height:
        rotated = imutils.rotate(img, angle=270)
    else:
        rotated = img.copy()

    return rotated


def sharpen_edge(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    dilated = cv2.dilate(blurred, rectKernel, iterations=2)
    edged = cv2.Canny(dilated, 75, 200, apertureSize=3)
    return edged


def binarize(img, threshold):
    threshold = np.mean(img)
    thresh, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(binary, rectKernel, iterations=2)
    return dilated


def find_receipt_bounding_box(binary, img):
    global rect

    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    largest_cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_cnt)
    box = np.intp(cv2.boxPoints(rect))
    boxed = cv2.drawContours(img.copy(), [box], 0, (0, 255, 0), 20)
    return boxed, largest_cnt


def find_tilt_angle(largest_contour):
    angle = rect[2]  # Find the angle of vertical line
    print("Angle_0 = ", round(angle, 1))
    if angle < -45:
        angle += 90
        print("Angle_1:", round(angle, 1))
    else:
        uniform_angle = abs(angle)
    print("Uniform angle = ", round(uniform_angle, 1))
    return rect, uniform_angle


def adjust_tilt(img, angle):
    if angle >= 5 and angle < 80:
        rotated_angle = 0
    elif angle < 5:
        rotated_angle = angle
    else:
        rotated_angle = 270+angle
    tilt_adjusted = imutils.rotate(img, rotated_angle)
    delta = 360-rotated_angle
    return tilt_adjusted, delta


def crop(img, largest_contour):
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped = img[y:y+h, x:x+w]
    return cropped




def enhance_txt(img):
    w = img.shape[1]
    h = img.shape[0]
    w1 = int(w*0.05)
    w2 = int(w*0.95)
    h1 = int(h*0.05)
    h2 = int(h*0.95)
    ROI = img[h1:h2, w1:w2]  # 95% of center of the image
    threshold = np.mean(ROI) * 0.98  # % of average brightness

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (1, 1), 0)
    edged = 255 - cv2.Canny(blurred, 100, 150, apertureSize=7)

    thresh, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

    return binary

@app.route('/')
def home():
    return "Welcome to the Image Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    # Get image buffer from request
    image_data = request.data
    image = Image.open(BytesIO(image_data))
    raw_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    rotated = orient_vertical(raw_img)
    edged = sharpen_edge(rotated)
    binary = binarize(edged, 100)
    boxed, largest_cnt = find_receipt_bounding_box(binary, rotated)
    boxed_rgb = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB)

    # Adjust tilt
    rect, angle = find_tilt_angle(largest_cnt)
    tilted, delta = adjust_tilt(boxed, angle)
    print(f"{round(delta,2)} degree adjusted towards right.")

    # Crop
    cropped = crop(tilted, largest_cnt)

    # Enhance txt
    enhanced = enhance_txt(cropped)
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

    # Bild auf dem Server speichern
    image_filename = os.path.join(IMAGE_SAVE_PATH, 'uploaded_image.png')
    enhanced_rgb.save(image_filename)

    # Preprocess the image
    pixel_values = processor(images=enhanced_rgb, return_tensors="pt").pixel_values.to(device)
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
