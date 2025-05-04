import os
import spacy
import pandas as pd
from flask import Flask, request, render_template, redirect
from utils import perform_ocrs_trocr
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

# TrOCR function 
def perform_ocr_trocr(img):
    pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert OpenCV image to PIL
    pixel_values = trocr_processor(images=pil_image, return_tensors="pt").pixel_values
    generated_ids = trocr_model.generate(pixel_values)
    generated_text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

# === CONFIGURATION ===
UPLOAD_FOLDER = os.path.join('static', 'uploads')
MODEL_OTHER = 'ner_model'
EXCEL_FILE = 'output.xlsx'
ALL_COLUMNS = ['Image', 'PATIENT_NAME', 'AGE', 'DATE', 'DOCTOR_NAME', 'ADDRESS', 'MEDICINES', 'MEDICINE', 'HOSPITAL_NAME', 'DIAGNOSIS']

# === INIT MODELS ===
nlp_other = spacy.load(MODEL_OTHER)

# === FLASK APP SETUP ===
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === SAVE TO EXCEL ===
def save_to_excel(image_name, entities):
    if os.path.exists(EXCEL_FILE):
        df = pd.read_excel(EXCEL_FILE)
    else:
        df = pd.DataFrame(columns=ALL_COLUMNS)

    row = {col: 'NA' for col in ALL_COLUMNS}
    row['Image'] = image_name
    for text, label in entities:
        if label in row:
            if row[label] == 'NA':
                row[label] = text
            else:
                row[label] += f", {text}"

    if image_name in df['Image'].values:
        df.loc[df['Image'] == image_name, list(row.keys())[1:]] = list(row.values())[1:]
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df.to_excel(EXCEL_FILE, index=False)

# === MORPHOLOGY + OCR + NER ===
def extract_entities_from_image(image_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    img = cv2.resize(img, (576, int(576 * h / w)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Morphological processing 
    kernel = np.ones((5, 5), np.uint8)
    linek = np.zeros((11, 11), dtype=np.uint8)
    linek[5, ...] = 1
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, linek, iterations=1)
    processed = cv2.subtract(gray, opened)
    _, thresh = cv2.threshold(processed, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    # bounding box detection
    contours, _ = cv2.findContours(dilated, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    extracted_text = perform_ocrs_trocr(image_path)

    doc_other = nlp_other(extracted_text)

    # Combine entities
    #medicine_entities = [(ent.text, ent.label_) for ent in doc_medicine.ents if ent.label_ == "MEDICINE"]
    other_entities = [(ent.text, ent.label_) for ent in doc_other.ents if ent.label_ != "MEDICINE"]

    return other_entities # + medicine_entities

# === ROUTES ===
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            entities = extract_entities_from_image(image_path)
            save_to_excel(filename, entities)

            return render_template('result.html', entities=entities, image_filename=filename)
    return render_template('index.html')

# === MAIN ===
if __name__ == '__main__':
    app.run(debug=True)
