from fastapi import FastAPI
from typing import List, Dict
from src.preprocessing import preprocessing_input
from src.model import predict_model
from typing import Union


# Initialize FastAPI
app = FastAPI()

# Prediction endpoint
@app.post("/predict")
async def predict(input_data: List[Dict[str, Union[str, float, int, None]]]):
    input_df = preprocessing_input(input_data)

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
