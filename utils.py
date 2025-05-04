# ocr_handler.py
import io
import os
from google.cloud import vision
from google.oauth2 import service_account

SERVICE_ACCOUNT_FILE = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "fallback.json")
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
client = vision.ImageAnnotatorClient(credentials=credentials)

def perform_ocrs_trocr(image_path):
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)

    extracted_text = ""
    if response.full_text_annotation:
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    paragraph_text = " ".join([
                        "".join([symbol.text for symbol in word.symbols])
                        for word in paragraph.words
                    ])
                    extracted_text += paragraph_text + "\n\n"
    return extracted_text
