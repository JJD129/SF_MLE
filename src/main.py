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
@app.get("/predict")
