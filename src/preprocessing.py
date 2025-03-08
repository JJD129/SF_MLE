import pandas as pd

import joblib
import os


# define artifacts dir
artifactsdir = os.path.join(os.getcwd(), "artifacts")

# artifact directories
imputer_path = os.path.join(artifactsdir, "imputer.pkl")
encoder_path = os.path.join(artifactsdir, "encoder.pkl")
features_path = os.path.join(artifactsdir, "selected_featurs.pkl")

# load preprocessing artifacts
imputer = joblib.load(imputer_path)
encoder = joblib.load(encoder_path)
selected_features = joblib.load(features_path)

def convert_money_and_percents(df):
    """
    converting x_75 and _x89 to float
    """

    if "x_75" in df.columns:
        df["x_75"] = df["x_75"].str.replace('$','',regex=False)\
                                .str.replace(',','',regex=False)\
                                .str.replace(')','',regex=False)\
                                .str.replace('(','-',regex=False)\
                                .astype(float)
    if "x_89" in df.columns:
        df["x_89"].str.repace('%','',regex=False)\
                    .astype(float)
    
    return df