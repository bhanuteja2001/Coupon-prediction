"""
MLflow Model Training and Registration Script

This script trains final XGBoost models using the best hyperparameters from previous experiments,
evaluates them on a test set, and registers the best model in MLflow.

It uses Prefect for workflow management and MLflow for experiment tracking and model registration.
"""

import os
import pickle
from datetime import date
from typing import Dict, List

import mlflow
import xgboost as xgb
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.metrics import roc_auc_score

# Configuration
VAL_EXPERIMENT_NAME = "mlops-project-coupon-accepting-experiment"
EXPERIMENT_NAME = "final-model-mlops-project-coupon-accepting"
TRACKING_SERVER_HOST = os.getenv('MLFLOW_SERVER_HOST')
TRACKING_SERVER_URI = f"http://{TRACKING_SERVER_HOST}:5000"
XGB_PARAMS_FLOAT = ['learning_rate', 'min_child_weight', 'reg_alpha', 'reg_lambda']
XGB_PARAMS_INT = ['max_depth', 'seed', 'verbosity']

mlflow.set_tracking_uri(TRACKING_SERVER_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

def load_pickle(filename: str) -> object:
    """Load a pickle file."""
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@task
def train_and_log_final_model(data_path: str, params: Dict[str, float]) -> None:
    """Train a final XGBoost model, evaluate it, and log results to MLflow."""
    x_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    x_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    with mlflow.start_run():
        mlflow.set_tag("model", "xgboost")
        mlflow.set_tag("eval", "test set")

        train = xgb.DMatrix(x_train, label=y_train)
        test = xgb.DMatrix(x_test, label=y_test)

        # Convert parameters to appropriate types
        for param in XGB_PARAMS_FLOAT:
            params[param] = float(params[param])
        for param in XGB_PARAMS_INT:
            params[param] = int(params[param])

        mlflow.log_params(params)

        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=500,
            evals=[(test, "test")],
            early_stopping_rounds=50,
        )

        pred_target = booster.predict(test)
        auc_score = roc_auc_score(y_test, pred_target)
        mlflow.log_metric("auc_score", auc_score)

        mlflow.xgboost.log_model(booster, artifact_path="final_models")
        print(f"Default artifacts URI: '{mlflow.get_artifact_uri()}'")

        markdown_auc_report = f"""# AUC Report TEST SET

        ## Summary

        Coupon Acceptance Prediction

        ## AUC XGBoost Model

        | Date       | AUC   |
        |:-----------|------:|
        | {date.today()} | {auc_score:.2f} |
        """

        create_markdown_artifact(
            key="coupon-model-report", markdown=markdown_auc_report
        )

@flow
def main_flow(top_n: int = 5) -> None:
    """Main flow to train final models and register the best one."""
    client = MlflowClient()

    # Get top N runs from validation experiment
    experiment = client.get_experiment_by_name(VAL_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        filter_string="tags.eval = 'validation set'",
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.auc_score DESC"],
    )

    # Train final models using top N configurations
    for run in runs:
        print(f"Training final model with run ID: {run.info.run_id}")
        train_and_log_final_model(data_path="others", params=run.data.params)

    # Select and register the best model
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.auc_score DESC"],
    )[0]

    best_model_uri = f"runs:/{best_run.info.run_id}/final_models"
    print(f"Best model URI: {best_model_uri}")
    
    mlflow.register_model(
        model_uri=best_model_uri,
        name="coupon-accepting-xgb-model",
    )

if __name__ == "__main__":
    TOP_NUMBER = 5
    main_flow(TOP_NUMBER)