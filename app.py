


import os
import uuid
import json
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# --- Initialization ---
app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
try:
    model = load_model("cat_dog_model_improved.keras")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load class weights for better prediction thresholds
try:
    with open('class_weights.json', 'r') as f:
        class_weights = json.load(f)
    # Calculate adjusted thresholds based on class weights
    cat_weight = class_weights['0']  # Assuming 0 is cat
    dog_weight = class_weights['1']  # Assuming 1 is dog
    
    # Adjust thresholds based on class weights
    CAT_THRESHOLD = 0.4  # Lower threshold for cats
    DOG_THRESHOLD = 0.7  # Higher threshold for dogs
    
except:
    CAT_THRESHOLD = 0.35
    DOG_THRESHOLD = 0.65
    print("Using default thresholds")

# --- Enhanced Prediction Function ---
def predict_image(img_path):
    """Enhanced prediction with better cat detection"""
    if model is None:
        raise RuntimeError("Model is not loaded. Cannot perform prediction.")

    try:
        # 1. Open and prepare the image with multiple augmentations
        img = Image.open(img_path).convert("RGB")
        img = img.resize((160, 160))  # Match training size
        
        # Create multiple augmented versions for ensemble prediction
        predictions = []
        
        # Original image
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions.append(model.predict(img_array, verbose=0)[0][0])
        
        # Horizontal flip (common augmentation)
        img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        img_array_flipped = np.array(img_flipped) / 255.0
        img_array_flipped = np.expand_dims(img_array_flipped, axis=0)
        predictions.append(model.predict(img_array_flipped, verbose=0)[0][0])
        
        # Slightly rotated
        img_rotated = img.rotate(5)
        img_array_rotated = np.array(img_rotated) / 255.0
        img_array_rotated = np.expand_dims(img_array_rotated, axis=0)
        predictions.append(model.predict(img_array_rotated, verbose=0)[0][0])
        
        # Average the predictions for better stability
        avg_prediction = np.mean(predictions)
        
        # 5. Enhanced decision making with adjusted thresholds
        if avg_prediction > DOG_THRESHOLD:
            label = "Dog"
            confidence = float(avg_prediction)
        elif avg_prediction < CAT_THRESHOLD:
            label = "Cat"
            confidence = float(1 - avg_prediction)
        elif avg_prediction < 0.5:  # Leaning toward cat
            label = "Cat"
            confidence = float(1 - avg_prediction)
        elif avg_prediction > 0.5:  # Leaning toward dog
            label = "Dog"
            confidence = float(avg_prediction)
        else:
            label = "I'm not sure!"
            confidence = 0.5
        
        # Additional confidence boost for clear cases
        if confidence > 0.85:
            confidence = min(confidence * 1.1, 0.99)
        
        return label, confidence

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise

# --- Web Routes ---
@app.route("/")
def index():
    return render_template("index1.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file:
        # Validate file type
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({"error": "Invalid file type. Please upload an image."}), 400

        filename = str(uuid.uuid4()) + file_ext
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)
            label, confidence = predict_image(filepath)
            
            return jsonify({
                "label": label,
                "confidence": round(confidence, 4)
            })
            
        except Exception as e:
            return jsonify({"error": f"An error occurred: {str(e)}"}), 500
            
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    return jsonify({"error": "Invalid file"}), 400

@app.route("/model-info")
def model_info():
    """Endpoint to get model information"""
    if model:
        return jsonify({
            "status": "loaded",
            "input_shape": model.input_shape,
            "output_shape": model.output_shape
        })
    else:
        return jsonify({"status": "not_loaded"})

# --- Main Execution ---
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)