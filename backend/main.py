from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import json

# 1. Initialize FastAPI
app = FastAPI(title="DermAI Backend API")

# 2. ENABLE CORS (Essential for React/Lovable to talk to Python)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Model Loading
MODEL_PATH = os.path.join('models', 'skin_cancer_model.h5')
model = tf.keras.models.load_model(MODEL_PATH)

# Clinical Database
DATA_INFO = {
    0: {'label': 'Actinic Keratoses', 'risk': 'High', 'type': 'Pre-Malignant'},
    1: {'label': 'Basal Cell Carcinoma', 'risk': 'Critical', 'type': 'Malignant'},
    2: {'label': 'Benign Keratosis', 'risk': 'Low', 'type': 'Benign'},
    3: {'label': 'Dermatofibroma', 'risk': 'Low', 'type': 'Benign'},
    4: {'label': 'Melanoma', 'risk': 'Critical', 'type': 'Malignant'},
    5: {'label': 'Nevus', 'risk': 'Low', 'type': 'Benign'},
    6: {'label': 'Vascular Lesions', 'risk': 'Low', 'type': 'Benign'}
}

# 4. Image Preprocessing Function
def prepare_image(image_data):
    img = Image.open(io.BytesIO(image_data)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# 5. The Prediction Endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 1. Read and Process the Image
        contents = await file.read()
        processed_image = prepare_image(contents)

        # 2. Run Neural Network Inference
        predictions = model.predict(processed_image)
        idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # Get data from your DATA_INFO list
        result = DATA_INFO[idx]

        # 3. Handle JSON Database Logic
        file_path = "patient_data.json"
        
        # Create the record object
        patient_record = {
            "classification": result['label'],
            "confidence": confidence,
            "risk": result['risk'],
            "type": result['type'],
            "timestamp": str(np.datetime64('now'))
        }

        # Load existing history or start fresh list
        history = []
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                try:
                    history = json.load(f)
                    # If the file exists but isn't a list yet, reset it
                    if not isinstance(history, list):
                        history = []
                except json.JSONDecodeError:
                    history = []

        # Add new result and save back to disk
        history.append(patient_record)
        with open(file_path, "w") as f:
            json.dump(history, f, indent=4)

        # 4. Return JSON response to the Frontend
        return {
            "success": True,
            "classification": result['label'],
            "confidence": f"{confidence * 100:.2f}%",
            "risk_level": result['risk'],
            "type": result['type']
        }

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return {
            "success": False, 
            "error": str(e)
        }
# Health check route
@app.get("/")
def home():
    return {"status": "Online", "model": "MobileNetV2 (93.25% Acc)"}