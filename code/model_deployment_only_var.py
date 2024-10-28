"""
Flask application for coupon acceptance prediction.

This module sets up a Flask server that loads an MLflow model and provides
an endpoint for making predictions on coupon acceptance.
"""

import os
from typing import Dict, Any

import mlflow
from flask import Flask, jsonify, request

import model_deployment_only_model as model

# Constants
RUN_ID = os.getenv('RUN_ID')
app = Flask('coupon-accepting-prediction')

def get_model_location(run_id: str) -> str:
    """
    Get the location of the model artifacts.

    Args:
        run_id: The MLflow run ID.

    Returns:
        The location of the model artifacts.
    """
    model_location = os.getenv('MODEL_LOCATION')
    if model_location:
        return model_location

    model_bucket = os.getenv('MODEL_BUCKET', 'mlflow-artifacts-mlops-project-storage')
    experiment_id = os.getenv('EXPERIMENT_ID', '2')

    return f's3://{model_bucket}/{experiment_id}/{run_id}/artifacts/final_models'

def load_model():
    """
    Load the MLflow model.

    Returns:
        The loaded MLflow model.
    """
    model_location = get_model_location(RUN_ID)
    return mlflow.pyfunc.load_model(model_location)

@app.route('/predict', methods=['POST'])
def predict_endpoint() -> Dict[str, Any]:
    """
    Endpoint for making predictions.

    Returns:
        A JSON response containing the prediction and model version.
    """
    ride = request.get_json()

    loaded_model = load_model()
    model_service = model.ModelService(loaded_model, RUN_ID)
    features = model_service.prepare_features(ride)
    pred = model_service.predict(features)

    return jsonify({
        'coupon_accepting': pred,
        'model_version': RUN_ID
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)