from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from src. preprocessing import preprocessing_input
from src.model import predict_model

# initalize FastAPI
app = FastAPI()

# define API input structure 
class InputData(BaseModel):
    features: Dict[str, str] # features is defined as key-value pairs [column names, value]

# pred endpoint
@app.post("/predict")
async def predict(input_data: List[InputData]):
    # convert input JSON to a dataframe
    input_dicts = [item.features for item in input_data]
    input_df = preprocessing_input(input_dicts)

    # run model pred
    probabilities, predictions = predict_model(input_df)

    # format output as JSON
    results = [
        {
            "business_outcome": int(pred),
            "prediction": int(prob),
            "feature_inputs": data

        }
        for data, pred, prob in zip(input_dicts, predictions, probabilities)
    ]
    return results