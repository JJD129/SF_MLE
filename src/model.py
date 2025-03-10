import joblib
import os


# define artifacts dir
artifactsdir = os.path.join(os.getcwd(), "artifacts")

# artifact directories
model_path = os.path.join(artifactsdir, "model.pkl")

# load preprocessing artifacts
model = joblib.load(model_path)

def predict_model(input_df):
    """
    runs trained model on preprocessed input data

    Args: 
        input_df (pd.DataFrame): preprocessed input features

    Returns:
        tuple: (probablitlies, predictions)
    """

    probabilities = model.predict_proba(input_df)
    predictions = (probabilities >= 0.75).astype(int)

    return(probabilities, predictions)   