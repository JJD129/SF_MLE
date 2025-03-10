import joblib
import os
import sys
import logging
from src.exception import CustomException

# define artifacts dir
artifactsdir = os.path.join(os.getcwd(), "artifacts")

# artifact directories
model_path = os.path.join(artifactsdir, "model.pkl")
features_path = os.path.join(artifactsdir, "selected_features.pkl")

# load preprocessing artifacts
model = joblib.load(model_path)
selected_features = joblib.load(features_path)

def predict_model(input_df):
    """
    runs trained model on preprocessed input data

    Args: 
        input_df (pd.DataFrame): preprocessed input features

    Returns:
        tuple: (probablitlies, predictions)
    """
    try:

        # only selected features are used
        input_df = input_df[selected_features]

        probabilities = model.predict_proba(input_df)
        predictions = (probabilities[:, 1] >= 0.75).astype(int)

        logging.info("Successfully made pred")

        return probabilities[:, 1], predictions
    
    except Exception as e:
        logging.error("Error in predict_model function", exc_info=True)
        raise CustomException(e, sys)