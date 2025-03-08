from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from src.preprocessing import preprocessing_input
from src.model import predict_model

# Initialize FastAPI
app = FastAPI()

# Define API input structure (List of Dictionaries)
class InputData(BaseModel):
    __root__: Dict[str, str]  # Each item is a dictionary with key-value pairs

# Prediction endpoint
@app.post("/predict")
async def predict(input_data: List[Dict[str, str]]):  # Accepts a list of dictionaries directly
    input_df = preprocessing_input(input_data)  # ✅ Preprocess input

    # Run model prediction
    probabilities, predictions = predict_model(input_df)

    # Format output as JSON
    results = [
        {
            "business_outcome": int(pred[0]),
            "prediction": float(prob[0]),
            "feature_inputs": data
        }
        for data, pred, prob in zip(input_data, predictions, probabilities)
    ]
    return results
