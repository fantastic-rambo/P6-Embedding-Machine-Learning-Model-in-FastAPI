from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import uvicorn
import os

app = FastAPI()

# Load your sepsis prediction model
model_path = r"C:\Users\IKE\OneDrive - Azubi Africa\Project1\Embedding-Machine-Learning-Model-in-FastAPI\model\best_model_rf.joblib"  # Replace with your actual model path
sepsis_model = joblib.load(model_path)

scaler = r"C:\Users\IKE\OneDrive - Azubi Africa\Project1\Embedding-Machine-Learning-Model-in-FastAPI\model\scaler.joblib"

class PatientData(BaseModel):
    PRG: float
    PL: float
    PR: float
    SK: float
    TS: float
    M11: float
    BD2: float
    Age: int
    Insurance: int


@app.post("/predict_sepsis")
async def predict_sepsis(data: PatientData):
    try:
        # Convert data to a format compatible with the model
        input_data = [[
            data.PRG, data.PL, data.PR, data.SK, data.TS, data.M11, data.BD2, data.Age, data.Insurance
        ]]
        
        # Make prediction
        prediction = sepsis_model.predict(input_data)

        # Return prediction as JSON response
        return JSONResponse(content={"prediction": int(prediction[0])}, status_code=200)

    except Exception as e:
        # Handle exceptions and return an error response
        return JSONResponse(content={"error": str(e)}, status_code=500)
