from fastapi import FastAPI
from pydantic import BaseModel, RootModel
from typing import List, Dict
from src. preprocessing import preprocessing_input
from src.model import predict_model

# initalize FastAPI
app = FastAPI()

# define API input structure 
class InputData(BaseModel):
    root: List[Dict[str, str]]

# pred endpoint
@app.post("/predict")
async def predict(input_data: InputData):
    # convert input JSON to a dataframe
    input_dicts = input_data.root
    input_df = preprocessing_input(input_dicts)

    # run model pred
    probabilities, predictions = predict_model(input_df)

    # format output as JSON
    results = [
        {
            "business_outcome": int(pred[0]),
            "prediction": float(prob[0]),
            "feature_inputs": data

        }
        for data, pred, prob in zip(input_dicts, predictions, probabilities)
    ]
    return results