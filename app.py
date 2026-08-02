import os
import io
import urllib.request
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import tensorflow as tf

MODEL_PATH  = "saved_model/plant_disease_model.h5"
LABELS_PATH = "saved_model/class_names.txt"
IMG_SIZE    = 224
TOP_K       = 3

if not os.path.exists(MODEL_PATH):
    print(" Downloading model from Google Drive...")
    os.makedirs("saved_model", exist_ok=True)
    url = "https://drive.google.com/uc?export=download&id=1tDfqodbnfOPhX0qoiXhgEozmY9MdZzXu&confirm=t"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Model downloaded!")

if not os.path.exists(LABELS_PATH):
    print("📥 Downloading class names...")
    url = "https://drive.google.com/uc?export=download&id=109P3HC2aWgfp6Pvg_YtkRNF530pSpXYs"
    urllib.request.urlretrieve(url, LABELS_PATH)
    print("Class names downloaded!")

print("🌿 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH) as f:
    CLASS_NAMES = [line.strip() for line in f.readlines()]
print(f"Model ready — {len(CLASS_NAMES)} classes")

app = Flask(__name__, static_folder=".")

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
    return {"plant": plant, "disease": disease, "healthy": healthy}

def preprocess_image(pil_img):
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    try:
        img = Image.open(io.BytesIO(file.read()))
        tensor = preprocess_image(img)
        preds = model.predict(tensor, verbose=0)[0]
        top_indices = np.argsort(preds)[::-1][:TOP_K]
        results = []
        for idx in top_indices:
            label_info = prettify_label(CLASS_NAMES[idx])
            results.append({**label_info, "confidence": float(preds[idx])})
        return jsonify({"predictions": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
