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

try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from '{MODEL_PATH}'")
except Exception as e:
    print(f"!!!!!!!!!! FATAL: Error loading model !!!!!!!!!!")
    print(f"ERROR_TYPE: {type(e)}")
    print(f"ERROR_DETAILS: {e}")
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    model = None

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
    print("--- PREDICT ENDPOINT HIT ---")

    if model is None:
        return {"error": "Model could not be loaded. Please check the server logs for the specific error."}

    try:
        print("--- STEP 1: Converting input data to DataFrame. ---")
        input_df = pd.DataFrame([data.dict()])
        print(f"--- STEP 2: DataFrame created successfully. ---")

        # --- ✅ FINAL TEST: We comment out the model prediction ---
        # We will return a hardcoded value instead.
        # If this request succeeds, the crash is on the model.predict() line.
        # If it still fails, the crash is on the pd.DataFrame() line.
        
        # prediction_raw = model.predict(input_df)[0]
        prediction_raw = 1 # Dummy value
        
        print(f"--- STEP 3: Using dummy prediction. Raw output: {prediction_raw} ---")

        species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        predicted_species = species_map.get(int(prediction_raw), "Unknown")
        
        response = {
            "prediction": int(prediction_raw),
            "predicted_species": predicted_species
        }
        print("--- STEP 4: Created response object. Returning success. ---")
        return response

    except Exception as e:
        print(f"!!!!!!!!!! CRITICAL ERROR during prediction !!!!!!!!!!")
        print(f"ERROR_TYPE: {type(e)}")
        print(f"ERROR_DETAILS: {e}")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return {"error": f"Prediction failed. Check server logs for details."}


# 5. Create a root endpoint for health checks
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Welcome to the Iris Prediction API!"}
