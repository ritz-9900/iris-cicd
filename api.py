# api.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# 1. Initialize FastAPI app
app = FastAPI(title="Iris Model API", description="API for predicting Iris species", version="1.0")

# 2. Load your model artifact
# --- CHANGE THIS SECTION ---
# We simplify the path. The Dockerfile will now place the model
# in the same directory as this script.
MODEL_PATH = 'model.joblib' 

try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from '{MODEL_PATH}'")
except FileNotFoundError:
    print(f"Error: Model file not found at '{MODEL_PATH}'. The API will not be able to make predictions.")
    model = None
# --- END OF CHANGES ---

# 3. Define the structure of the input data for validation
class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# 4. Create the prediction endpoint
@app.post("/predict")
def predict(data: IrisData):
    """Takes Iris flower measurements and returns the predicted species."""
    if model is None:
        return {"error": "Model not loaded. Please check the server logs."}

    # Convert the input into a pandas DataFrame, which the scikit-learn model expects
    input_df = pd.DataFrame([data.dict()])
    
    # Make the prediction
    prediction_raw = model.predict(input_df)[0]
    
    # Map the numeric output to a human-readable class name
    species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    predicted_species = species_map.get(int(prediction_raw), "Unknown")
    
    return {
        "prediction": int(prediction_raw),
        "predicted_species": predicted_species
    }

# 5. Create a root endpoint for health checks
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Welcome to the Iris Prediction API!"}
