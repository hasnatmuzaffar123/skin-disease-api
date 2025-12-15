import uvicorn
import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SKIN_MODEL_PATH = os.path.join(BASE_DIR, "best_model_finetuned.keras")
PNEUMONIA_MODEL_PATH = os.path.join(BASE_DIR, "pneumonia_model_finetuned.keras")

SKIN_CLASSES = ['Chickenpox', 'Eczema', 'HFMD', 'Healthy', 'Jaundice', 'Ringworm', 'Scabies']
PNEUMONIA_CLASSES = ['Normal', 'Pneumonia']


app = FastAPI(
    title="Symptom Analysis Using AI Model",
    description="API for detecting skin diseases using images and pneumonia from X-RAY images",
    version="2.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


skin_model = None
pneumonia_model = None


@app.on_event("startup")
async def load_models():
    global skin_model, pneumonia_model
    print("=" * 50)
    print("🔍 Loading Models...")
    print("=" * 50)
    
    
    print(f"📋 Skin Model Path: {SKIN_MODEL_PATH}")
    if not os.path.exists(SKIN_MODEL_PATH):
        print(" Skin model NOT FOUND!")
    else:
        try:
            skin_model = tf.keras.models.load_model(SKIN_MODEL_PATH, compile=False)
            print(" Skin model loaded successfully!")
            print(f"   Input Shape: {skin_model.input_shape}")
        except Exception as e:
            print(f" ERROR loading skin model: {e}")
    
    
    print(f"\n📋 Pneumonia Model Path: {PNEUMONIA_MODEL_PATH}")
    if not os.path.exists(PNEUMONIA_MODEL_PATH):
        print(" Pneumonia model NOT FOUND!")
    else:
        try:
            pneumonia_model = tf.keras.models.load_model(PNEUMONIA_MODEL_PATH, compile=False)
            print(" Pneumonia model loaded successfully!")
            print(f"   Input Shape: {pneumonia_model.input_shape}")
        except Exception as e:
            print(f" ERROR loading pneumonia model: {e}")
    
    print("=" * 50)


def preprocess_image(image_bytes, input_size=224):
    """
    Converts raw image bytes to a Numpy array compatible with the model.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        img = img.resize((input_size, input_size))
        
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        return img_array
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")



@app.get("/")
async def root():
    return {
        "message": "Medical Diagnosis Prediction API is running",
        "version": "2.0.0",
        "available_endpoints": {
            "docs": "/docs",
            "skin_disease_predict": "/predict/skin",
            "pneumonia_predict": "/predict/pneumonia",
            "health": "/health",
            "classes": "/classes"
        },
        "status": "online"
    }

@app.get("/health")
async def health_check():
    """Checks if models are loaded and server is running."""
    return {
        "status": "healthy" if (skin_model is not None or pneumonia_model is not None) else "error",
        "skin_model": "loaded" if skin_model is not None else "not_loaded",
        "pneumonia_model": "loaded" if pneumonia_model is not None else "not_loaded"
    }

@app.get("/classes")
async def get_classes():
    """Returns the list of disease classes for both models."""
    return {
        "skin_disease_classes": {
            "classes": SKIN_CLASSES,
            "num_classes": len(SKIN_CLASSES)
        },
        "pneumonia_classes": {
            "classes": PNEUMONIA_CLASSES,
            "num_classes": len(PNEUMONIA_CLASSES)
        }
    }

@app.post("/predict/skin")
async def predict_skin(file: UploadFile = File(...)):
    """
    Endpoint to predict skin disease from an uploaded image.
    
    Request:
        - file: Image file (JPG, JPEG, PNG)
    
    Response:
        - predicted_disease: The detected disease class
        - confidence: Confidence percentage (0-100)
        - all_probabilities: All class probabilities
    """
    if skin_model is None:
        raise HTTPException(status_code=503, detail="Skin model is not loaded. Check server logs.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}. Please upload an image.")

    try:
        contents = await file.read()
        processed_image = preprocess_image(contents)
        predictions = skin_model.predict(processed_image, verbose=0)
        
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx]) * 100
        predicted_class = SKIN_CLASSES[predicted_idx]
        
        all_probs = {
            class_name: float(prob) * 100 
            for class_name, prob in zip(SKIN_CLASSES, predictions[0])
        }
        
        return {
            "model_type": "skin_disease",
            "predicted_disease": predicted_class,
            "confidence": round(confidence, 2),
            "all_probabilities": {k: round(v, 2) for k, v in all_probs.items()}
        }

    except Exception as e:
        print(f"Skin Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/predict/pneumonia")
async def predict_pneumonia(file: UploadFile = File(...)):
    """
    Endpoint to predict pneumonia from a chest X-ray image.
    
    Request:
        - file: Image file (JPG, JPEG, PNG)
    
    Response:
        - prediction: Normal or Pneumonia
        - confidence: Confidence percentage (0-100)
        - all_probabilities: All class probabilities
    """
    if pneumonia_model is None:
        raise HTTPException(status_code=503, detail="Pneumonia model is not loaded. Check server logs.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}. Please upload an image.")

    try:
        contents = await file.read()
        processed_image = preprocess_image(contents)
        predictions = pneumonia_model.predict(processed_image, verbose=0)
        
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx]) * 100
        predicted_class = PNEUMONIA_CLASSES[predicted_idx]
        
        all_probs = {
            class_name: float(prob) * 100 
            for class_name, prob in zip(PNEUMONIA_CLASSES, predictions[0])
        }
        
        return {
            "model_type": "pneumonia",
            "prediction": predicted_class,
            "confidence": round(confidence, 2),
            "all_probabilities": {k: round(v, 2) for k, v in all_probs.items()}
        }

    except Exception as e:
        print(f"Pneumonia Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)