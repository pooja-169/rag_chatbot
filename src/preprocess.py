# src/preprocess.py

import os
import PyPDF2

def extract_text_from_pdf(file_path):
    pdf_text = ""
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            pdf_text += page.extract_text()
    return pdf_text

def chunk_text(text, chunk_size=300):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

if __name__ == "__main__":
    pdf_path = "data/example.pdf"
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    for c in chunks:
        print(c)
