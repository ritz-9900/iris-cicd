# api.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# 1. Initialize FastAPI app
app = FastAPI(title="Iris Model API", description="API for predicting Iris species", version="1.0")

# 2. Load your model artifact
MODEL_PATH = 'model.joblib' 

# --- ✅ FINAL CHANGE: Catch ANY exception during model loading ---
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from '{MODEL_PATH}'")
except Exception as e:
    # This will catch any error (version mismatch, corrupted file, etc.)
    # and print it to the logs.
    print(f"!!!!!!!!!! FATAL: Error loading model !!!!!!!!!!")
    print(f"ERROR_TYPE: {type(e)}")
    print(f"ERROR_DETAILS: {e}")
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
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
        # This message will now be accurate. The real error is in the logs.
        return {"error": "Model could not be loaded. Please check the server logs for the specific error."}

    try:
        # Convert the input into a pandas DataFrame
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
    except Exception as e:
        # Also add better error handling to the predict function itself
        print(f"!!!!!!!!!! ERROR during prediction !!!!!!!!!!")
        print(f"ERROR_TYPE: {type(e)}")
        print(f"ERROR_DETAILS: {e}")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return {"error": f"Prediction failed. Check server logs for details."}


# 5. Create a root endpoint for health checks
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Welcome to the Iris Prediction API!"}
