from fastapi import FastAPI
from typing import List, Dict
from src.preprocessing import preprocessing_input
from src.model import predict_model
from typing import Union
from src.logger import logging
from src.exception import CustomException
import joblib
import os
import sys

# Load selected features from the artifacts directory
artifacts_dir = "artifacts"
selected_features_path = os.path.join(artifacts_dir, "selected_features.pkl")
selected_features = joblib.load(selected_features_path)  # Load selected features list

# Initialize FastAPI
app = FastAPI()

# Prediction endpoint
@app.post("/predict")
async def predict(input_data: List[Dict[str, Union[str, float, int, None]]]):
    try:
        logging.info("Received request with input data: %s")
        input_df = preprocessing_input(input_data)

        # Run model prediction
        probabilities, predictions = predict_model(input_df)

        # Format output as JSON
        results = [
            {
                "business_outcome": int(pred),
                "prediction": float(prob),
                "feature_inputs": {key: float(input_df.iloc[i][key]) for key in selected_features if key in input_df.columns}
            }
            for i, (pred, prob) in enumerate(zip(predictions, probabilities))
        ]
        return results
    except Exception as e:
        logging.error(f"Error in API request: {str(e)}", exc_info=True)
        raise CustomException(e, sys)
