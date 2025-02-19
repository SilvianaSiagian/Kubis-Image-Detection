from flask import Flask, request, jsonify, render_template
import pickle
from PIL import Image
import numpy as np
import os

# Load the trained model
model_path = "model/kubis_cnn_model.pkl"
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Initialize the Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")

# Helper function to preprocess the image
def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = np.array(image)
    if len(image.shape) == 3 and image.shape[2] == 3:  # Check if the image is RGB
        image = image / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    else:
        return None

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    try:
        # Load the image
        image = Image.open(file)

        # Preprocess the image
        processed_image = preprocess_image(image, target_size=(224, 224))  # Adjust size if necessary
        if processed_image is None:
            return jsonify({"error": "Invalid image format. Please upload an RGB image."}), 400

        # Predict using the model
        predictions = model.predict(processed_image)
        class_names = ["Bukan Kubis", "Kubis Sehat", "Kubis Tidak Sehat"]
        predicted_class = class_names[np.argmax(predictions)]

        # Return the prediction
        return jsonify({"prediction": predicted_class})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
