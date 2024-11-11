import joblib
import pandas as pd

# Load trained model
model = joblib.load('../models/accident_severity_model.pkl')

def predict_severity(features):
    features_df = pd.DataFrame([features])
    return model.predict(features_df)
