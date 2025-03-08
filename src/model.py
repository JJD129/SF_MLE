import pandas as pd
import numpy as np

import joblib
import os


# define artifacts dir
artifactsdir = os.path.join(os.getcwd(), "artifacts")

# artifact directories
model_path = os.path.join(artifactsdir, "model.pkl")

# load preprocessing artifacts
model = joblib.load(model_path)