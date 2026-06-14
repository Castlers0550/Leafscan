"""
LeafScan AI - Hugging Face Gradio App
======================================
Upload this as app.py to your Hugging Face Space.
"""

import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf
import os

MODEL_PATH  = "plant_disease_model.h5"
LABELS_PATH = "class_names.txt"
IMG_SIZE    = 224

# Load model
print("🌿 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH) as f:
    CLASS_NAMES = [line.strip() for line in f.readlines()]
print(f"✅ Ready — {len(CLASS_NAMES)} classes")


def prettify_label(raw):
    raw_clean = raw.replace("_", " ")
    if "___" in raw:
        parts = raw.split("___")
        plant   = parts[0].replace("_", " ").strip()
        disease = parts[1].replace("_", " ").strip()
    else:
        plant   = raw_clean
        disease = raw_clean
    healthy = "healthy" in disease.lower()
    return plant, disease, healthy


def predict(image):
    if image is None:
        return "Please upload an image."

    img = Image.fromarray(image).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]
    top3  = np.argsort(preds)[::-1][:3]

    result = ""
    for i, idx in enumerate(top3):
        plant, disease, healthy = prettify_label(CLASS_NAMES[idx])
        confidence = preds[idx] * 100
        status = "✅ Healthy" if healthy else "⚠️ Disease detected"
        result += f"**#{i+1} — {plant}**\n"
        result += f"{status}: {disease}\n"
        result += f"Confidence: {confidence:.1f}%\n\n"

    return result.strip()


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload a leaf photo"),
    outputs=gr.Markdown(label="Results"),
    title="🌿 LeafScan AI — Plant Disease Detector",
    description="Upload a photo of a plant leaf and the AI will identify diseases instantly. Trained on 54,000+ images across 36 plant disease classes.\n\n*By Eben Siyabalapitiya*",
    examples=[],
    theme=gr.themes.Base(),
)

demo.launch()
