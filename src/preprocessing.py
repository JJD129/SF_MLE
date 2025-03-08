import pandas as pd
import numpy as np
import joblib
import os


# define artifacts dir
artifactsdir = os.path.join(os.getcwd(), "artifacts")

# artifact directories
imputer_path = os.path.join(artifactsdir, "imputer.pkl")
encoder_path = os.path.join(artifactsdir, "encoder.pkl")
features_path = os.path.join(artifactsdir, "selected_features.pkl")

# load preprocessing artifacts
imputer = joblib.load(imputer_path)
encoders = joblib.load(encoder_path)
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
        df["x_89"].str.replace('%','',regex=False)\
                    .astype(float)
    
    return df

def impute_missing_numerical(df):
    """
    simple mean imputation on mean on numerical cols
    """

    expected_features = imputer.feature_names_in_

    for col in expected_features:
        if col not in df.columns:
            df[col] = np.nan

    df[expected_features] = imputer.transform(df[expected_features])
    
    return df

def transform_ohe(df):
    """
    apply one hot encoding on pretrained encoders
    """

    df_transformed = df.copy()

    for var, encoder in encoders.items():
        if var in df.columns:
            encoder_df = pd.DataFrame(encoder.transform(df[[var]]), columns=encoder.get_feature_names_out([var]))
            df_transformed = pd.concat([df_transformed.drop(columns=[var]), encoder_df], axis=1)

    return df

def preprocessing_input(data):
    """
    preprocessing pipeline by applying all transformations
    """
    df = pd.DataFrame(data)
    df = convert_money_and_percents(df)
    df = impute_missing_numerical(df)
    df = transform_ohe(df)

    # return selected features
    return df[selected_features]

