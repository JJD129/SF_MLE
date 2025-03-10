import pandas as pd
import numpy as np
import joblib
from src.logger import logging
from src.exception import CustomException
import sys
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

# handle categoreical variables explicity
categorical_vars = ['x_16', 'x_51', 'x_65', 'x_9']

def convert_money_and_percents(df):
    """
    converting x_75 and _x89 to float
    """
    try:

        if "x_75" in df.columns:
            df["x_75"] = df["x_75"].str.replace('$','',regex=False)\
                                    .str.replace(',','',regex=False)\
                                    .str.replace(')','',regex=False)\
                                    .str.replace('(','-',regex=False)\
                                    .astype(float)
        if "x_89" in df.columns:
            df["x_89"] = df["x_89"].str.replace('%','',regex=False)\
                        .astype(float)
        
        logging.info("Successfully converted money and percent values")
        
        return df
    
    except Exception as e:
        logging.error("Error in convert_money_and_percents function", exc_info=True)
        raise CustomException(e, sys)


def impute_missing_numerical(df):
    """
    simple mean imputation on mean on numerical cols
    """
    try:

        expected_features = imputer.feature_names_in_

        for col in expected_features:
            if col not in df.columns:
                df[col] = np.nan

        df[expected_features] = imputer.transform(df[expected_features])
        
        logging.info("Successfully imputed missing numerical values")

        return df
    
    except Exception as e:
        logging.error("Error in impute_missing_numerical function", exc_info=True)
        raise CustomException(e, sys)

def transform_ohe(df):
    """
    apply one hot encoding on pretrained encoders
    """
    try:

        df_transformed = df.copy()

        for var, encoder in encoders.items():
            if var in categorical_vars and var in df.columns:
                encoder_df = pd.DataFrame(encoder.transform(df[[var]]), 
                                        columns=encoder.get_feature_names_out([var]))
                df_transformed = pd.concat([df_transformed.drop(columns=[var]), encoder_df], axis=1)
        
        logging.info("Successfully applied one-hot encoding")
        
        return df_transformed
    
    except Exception as e:
        logging.error("Error in transform_ohe function", exc_info=True)
        raise CustomException(e, sys)

def preprocessing_input(data):
    """
    preprocessing pipeline by applying all transformations
    """
    try:
        df = pd.DataFrame(data)
        df = convert_money_and_percents(df)
        df = impute_missing_numerical(df)
        df = transform_ohe(df)

        # Debugging: print the transformed column names before feature selection
        logging.info("Columns after OHE:", df.columns.tolist())

        df = df.reindex(columns=selected_features, fill_value = 0)
        # Debugging: print final selected features
        logging.info("Final selected features in df:", df.columns.tolist())

        logging.info("Preprocessing completed successfully.")
        
        return df
    
    except Exception as e:
        logging.error("Error in preprocessing_input function", exc_info=True)
        raise CustomException(e, sys)