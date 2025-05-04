from flask import Flask, request, jsonify, render_template, send_from_directory
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import tensorflow as tf
import os

app = Flask(__name__)

# Ensure uploads folder exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models
paddy_model = load_model("model/classifier.h5", compile=False)
disease_model = load_model("model/final_model.keras", compile=False)

# Disease-wise pesticide recommendations
pesticide_recommendations = {
    'bacterial_leaf_blight': {
        'chemical': 'Streptocycline (100 ppm) + Copper oxychloride (0.3%)',
        'organic': 'Spray neem oil (3%) and cow dung decoction as preventive measure'
    },
    'bacterial_leaf_streak': {
        'chemical': 'Streptocycline + Copper hydroxide (0.2%)',
        'organic': 'Use Panchagavya or Jeevamrutham sprays to improve immunity'
    },
    'bacterial_panicle_blight': {
        'chemical': 'Improve drainage, avoid excess nitrogen, apply recommended fungicides',
        'organic': 'Foliar application of Trichoderma-based biocontrol agents'
    },
    'blast': {
        'chemical': 'Tricyclazole @ 0.6 g/l or Isoprothiolane @ 1.5 ml/l',
        'organic': 'Use Pseudomonas fluorescens (10 g/l) as foliar spray'
    },
    'brown_spot': {
        'chemical': 'Mancozeb @ 2.5 g/l or Carbendazim @ 1 g/l',
        'organic': 'Neem leaf extract spray (5%) and compost tea foliar spray'
    },
    'dead_heart': {
        'chemical': 'Apply Carbofuran granules or Chlorantraniliprole',
        'organic': 'Release Trichogramma chilonis parasitoids and use neem cake'
    },
    'downy_mildew': {
        'chemical': 'Metalaxyl + Mancozeb @ 2 g/l',
        'organic': 'Use Bacillus subtilis-based bio-fungicides'
    },
    'hispa': {
        'chemical': 'Chlorpyrifos 20EC @ 2 ml/l or Quinalphos 25EC @ 2 ml/l',
        'organic': 'Neem seed kernel extract (NSKE 5%) spray'
    },
    'normal': {
        'chemical': 'No pesticide needed. Crop is healthy.',
        'organic': 'No treatment necessary. Maintain good field hygiene.'
    },
    'tungro': {
        'chemical': 'Control green leafhopper using Imidacloprid @ 0.5 ml/l',
        'organic': 'Introduce natural predators like Cyrtorhinus lividipennis'
    }
}
@app.route("/")

@app.route("/home")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    image_file = Image.open(file.stream).convert("RGB")

    # Save uploaded image
    filename = file.filename
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    image_file.save(save_path)

    # Step 1: Paddy or Not Paddy
    paddy_img = image_file.resize((128, 128))
    paddy_array = np.expand_dims(np.array(paddy_img) / 255.0, axis=0)
    paddy_prediction = paddy_model.predict(paddy_array)[0][0]
    is_paddy = paddy_prediction >= 0.5

    if not is_paddy:
        return jsonify({
            "label": "not_paddy_leaf",
            "confidence": float(1 - paddy_prediction),
            "message": "The uploaded image is not a paddy leaf.",
            "image_url": f"/uploads/{filename}"
        })

    # Step 2: Disease Classification
    disease_img = image_file.resize((256, 256))
    disease_array = np.expand_dims(np.array(disease_img) / 255.0, axis=0)

    disease_prediction = disease_model.predict(disease_array)
    disease_probabilities = tf.nn.softmax(disease_prediction[0]).numpy()
    disease_class = np.argmax(disease_probabilities)

    disease_labels = [
        'bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight',
        'blast', 'brown_spot', 'dead_heart', 'downy_mildew', 'hispa', 'normal', 'tungro'
    ]
    disease_label = disease_labels[disease_class]

    recommendation = pesticide_recommendations.get(disease_label, {
        'chemical': "No chemical recommendation available",
        'organic': "No organic recommendation available"
    })

    return jsonify({
        "label": disease_label,
        "chemical": recommendation["chemical"],
        "organic": recommendation["organic"],
        "image_url": f"/uploads/{filename}"
    })

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
