"""
Flask server for coupon acceptance prediction.

This module loads an MLflow model and provides an endpoint for making predictions
on coupon acceptance based on input features.
"""

import os
import pickle
from typing import Dict, Any

import mlflow
import pandas as pd
import xgboost as xgb
from flask import Flask, jsonify, request

# Constants
RUN_ID = os.getenv('RUN_ID')
MODEL_URI = f's3://mlflow-artifacts-mlops-project-storage/2/{RUN_ID}/artifacts/final_models'
MODEL = mlflow.pyfunc.load_model(MODEL_URI)

app = Flask('coupon-accepting-prediction')

def load_pickle(filename: str) -> Any:
    """Load a pickle file."""
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

def prepare_features(coupon_rec: Dict[str, Any]) -> pd.DataFrame:
    """Prepare features for prediction from raw coupon recommendation data."""
    columns = [
        'destination', 'weather', 'time', 'coupon', 'expiration',
        'same_direction', 'coupon_accepting'
    ]

    temp_df = pd.DataFrame(coupon_rec)
    temp_df.rename(
        columns={'direction_same': 'same_direction', 'Y': 'coupon_accepting'},
        inplace=True,
    )

    return temp_df[columns]

def convert_features_to_dmatrix(dataframe: pd.DataFrame) -> xgb.DMatrix:
    """Convert a pandas DataFrame to an XGBoost DMatrix."""
    features = [
        'destination', 'weather', 'time', 'coupon', 'expiration', 'same_direction'
    ]

    dict_vector = load_pickle('final_preprocessor.b')

    temp_dicts = dataframe[features].to_dict(orient="records")
    tf_temp_dicts = dict_vector.transform(temp_dicts)
    target = dataframe['coupon_accepting'].values

    return xgb.DMatrix(tf_temp_dicts, label=target)

def predict(features: xgb.DMatrix) -> str:
    """Predict coupon acceptance (Yes/No)."""
    print(f"Input type: {type(features)}")
    
    # Note: This is a placeholder implementation. In a real scenario,
    # you would use MODEL.predict(features) instead of a constant value.
    preds = 0.9
    
    return 'Yes' if preds > 0.5 else 'No'

@app.route('/predict', methods=['POST'])
def predict_endpoint() -> Dict[str, Any]:
    """Endpoint for making predictions."""
    ride = request.get_json()

    features = prepare_features(ride)
    pred = predict(features)

    return jsonify({
        'coupon_accepting': pred,
        'model_version': RUN_ID
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)