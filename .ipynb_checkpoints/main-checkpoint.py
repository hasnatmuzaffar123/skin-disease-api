import uvicorn
import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model_finetuned.keras")

CLASS_NAMES = ['Chickenpox', 'Eczema', 'HFMD', 'Healthy', 'Jaundice', 'Ringworm', 'Scabies']

# Initialize FastAPI app
app = FastAPI(
    title="Skin Disease Prediction API",
    description="API for detecting skin diseases from images",
    version="1.0.0"
)

# --- CORS MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for frontend access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None

# --- STARTUP EVENT: LOAD MODEL ---
@app.on_event("startup")
async def load_model():
    global model
    print("=" * 50)
    print(f"🔍 Looking for model at: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print("❌ ERROR: Model file NOT FOUND!")
        print(f"   Please ensure 'best_model_finetuned.keras' is in: {BASE_DIR}")
        return

    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded successfully!")
        print(f"   Model Input Shape: {model.input_shape}")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Failed to load model.")
        print(f"   Error details: {e}")
    print("=" * 50)

# --- HELPER: PREPROCESS IMAGE ---
def preprocess_image(image_bytes):
    """
    Converts raw image bytes to a Numpy array compatible with the model.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        img = img.resize((224, 224))
        
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        return img_array
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")

# --- ENDPOINTS ---

@app.get("/")
async def root():
    return {
        "message": "Skin Disease Prediction API is running",
        "docs_url": "/docs",
        "status": "online"
    }

@app.get("/health")
async def health_check():
    """Checks if the model is loaded and server is running."""
    if model is None:
        return {"status": "error", "message": "Model not loaded"}
    return {"status": "healthy", "message": "Model is ready"}

@app.get("/classes")
async def get_classes():
    """Returns the list of disease classes the model can predict."""
    return {"classes": CLASS_NAMES, "num_classes": len(CLASS_NAMES)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to predict skin disease from an uploaded image.
    
    Request:
        - file: Image file (JPG, JPEG, PNG)
    
    Response:
        - predicted_disease: The detected disease class
        - confidence: Confidence percentage (0-100)
        - all_probabilities: All class probabilities
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check server logs.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}. Please upload an image.")

    try:
        contents = await file.read()
        processed_image = preprocess_image(contents)
        predictions = model.predict(processed_image, verbose=0)
        
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx]) * 100
        predicted_class = CLASS_NAMES[predicted_idx]
        
        all_probs = {
            class_name: float(prob) * 100 
            for class_name, prob in zip(CLASS_NAMES, predictions[0])
        }
        
        return {
            "predicted_disease": predicted_class,
            "confidence": round(confidence, 2),
            "all_probabilities": {k: round(v, 2) for k, v in all_probs.items()}
        }

    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# --- RUN SERVER ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)