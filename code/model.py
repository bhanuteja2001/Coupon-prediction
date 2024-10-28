# pylint: disable=import-error, ungrouped-imports, invalid-name, line-too-long, too-many-locals, too-many-arguments
"""
Model Training Pipeline

This script implements a complete machine learning pipeline for coupon acceptance prediction.
It includes data loading, preprocessing, model training, and evaluation using MLflow for experiment tracking.
"""

import os
import pickle
import argparse
from pathlib import Path
from datetime import date

import numpy as np
import scipy
import mlflow
import pandas as pd
import sklearn
import xgboost as xgb
from prefect import flow, task
from sklearn.metrics import roc_auc_score
from prefect.artifacts import create_markdown_artifact
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

# Constants
EXPERIMENT_NAME = "mlops-project-coupon-accepting-experiment"
TRACKING_SERVER_HOST = "ec2-13-51-56-182.eu-north-1.compute.amazonaws.com"  # TODO: Use environment variable for this
TRACKING_SERVER_URI = f"http://{TRACKING_SERVER_HOST}:5000"
DATA_BUCKET = "state-persist"

@task(retries=3, retry_delay_seconds=2)
def read_data(filename: str) -> pd.DataFrame:
    """
    Read and preprocess the coupon recommendation data.

    Args:
        filename (str): Path to the input CSV file (compressed).

    Returns:
        pd.DataFrame: Preprocessed DataFrame with selected features.
    """
    # Read the CSV file
    coupon_rec_df = pd.read_csv(filename, compression='zip')

    # Rename columns for consistency
    coupon_rec_df.rename(
        columns={'direction_same': 'same_direction', 'Y': 'coupon_accepting'},
        inplace=True,
    )

    # Define feature categories
    categorical = ['destination', 'weather', 'time', 'coupon', 'expiration']
    boolean = ['same_direction', 'coupon_accepting']

    # Convert categorical columns to string type for memory efficiency
    coupon_rec_df[categorical] = coupon_rec_df[categorical].astype(str)

    # Return only the selected features
    return coupon_rec_df[categorical + boolean]

@task
def prepare_data_valid_set(whole_df: pd.DataFrame) -> tuple:
    """
    Prepare data for training and validation.

    Args:
        whole_df (pd.DataFrame): The entire dataset.

    Returns:
        tuple: Contains train and validation features and targets, DictVectorizer, and full train and test sets.
    """
    # Split data into train, validation, and test sets
    df_full_train, df_test = train_test_split(whole_df, test_size=0.2, random_state=42)
    df_train, df_val = train_test_split(df_full_train, test_size=0.2, random_state=42)

    categorical = ['destination', 'weather', 'time', 'coupon', 'expiration']
    boolean = ['same_direction']

    # Initialize and fit DictVectorizer
    dict_vector = DictVectorizer()

    # Transform train data
    train_dicts = df_train[categorical + boolean].to_dict(orient="records")
    X_train = dict_vector.fit_transform(train_dicts)

    # Transform validation data
    val_dicts = df_val[categorical + boolean].to_dict(orient="records")
    X_val = dict_vector.transform(val_dicts)

    # Extract target variables
    y_train = df_train['coupon_accepting'].values
    y_val = df_val['coupon_accepting'].values

    return X_train, X_val, y_train, y_val, dict_vector, df_full_train, df_test

@task(log_prints=True)
def train_best_model(
    train_features: scipy.sparse._csr.csr_matrix,
    val_features: scipy.sparse._csr.csr_matrix,
    train_target: np.ndarray,
    val_target: np.ndarray,
    dict_vector: sklearn.feature_extraction.DictVectorizer,
    learning_rate: float,
    min_child_weight: float,
    max_depth: int,
    reg_lambda: float,
    reg_alpha: float,
):
    """
    Train the best XGBoost model and log results with MLflow.

    Args:
        train_features, val_features: Training and validation features.
        train_target, val_target: Training and validation targets.
        dict_vector: Fitted DictVectorizer for feature transformation.
        learning_rate, min_child_weight, max_depth, reg_lambda, reg_alpha: Model hyperparameters.
    """
    with mlflow.start_run():
        # Set MLflow tags
        mlflow.set_tag("model", "xgboost")
        mlflow.set_tag("eval", "validation set")

        # Prepare DMatrix for XGBoost
        train = xgb.DMatrix(train_features, label=train_target)
        valid = xgb.DMatrix(val_features, label=val_target)

        # Save preprocessor
        Path("others").mkdir(exist_ok=True)
        with open("others/val_preprocessor.b", "wb") as f_out:
            pickle.dump(dict_vector, f_out)
        mlflow.log_artifact("others/val_preprocessor.b", artifact_path="val_preprocessor")

        # Set model parameters
        params = {
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "min_child_weight": min_child_weight,
            "objective": "binary:logistic",
            "reg_alpha": reg_lambda,
            "reg_lambda": reg_alpha,
            "seed": 42,
            'verbosity': 1,
        }

        # Log hyperparameters
        mlflow.log_params(params)

        # Train the model
        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=500,
            evals=[(valid, "validation")],
            early_stopping_rounds=50,
        )

        # Evaluate the model
        pred_target = booster.predict(valid)
        auc_score = roc_auc_score(val_target, pred_target)
        mlflow.log_metric("auc_score", auc_score)

        # Save the model
        mlflow.xgboost.log_model(booster, artifact_path="val_models")
        print(f"Default artifacts URI: '{mlflow.get_artifact_uri()}'")

        # Create and log markdown report
        markdown_auc_report = f"""# AUC Report VALIDATION SET

        ## Summary

        Coupon Acceptance Prediction

        ## AUC XGBoost Model

        | Date       | AUC   |
        |:-----------|------:|
        | {date.today()} | {auc_score:.2f} |
        """

        create_markdown_artifact(key="coupon-model-report", markdown=markdown_auc_report)

def dump_pickle(obj, filename: str):
    """Utility function to save an object as a pickle file."""
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)

@task
def prepare_data_test_set(df_train, df_test):
    """
    Prepare the final training and test datasets.

    Args:
        df_train (pd.DataFrame): Full training dataset.
        df_test (pd.DataFrame): Test dataset.
    """
    categorical = ['destination', 'weather', 'time', 'coupon', 'expiration']
    boolean = ['same_direction']

    # Initialize and fit DictVectorizer
    dict_vector = DictVectorizer()

    # Transform train data
    train_dicts = df_train[categorical + boolean].to_dict(orient="records")
    X_train = dict_vector.fit_transform(train_dicts)

    # Transform test data
    test_dicts = df_test[categorical + boolean].to_dict(orient="records")
    X_test = dict_vector.transform(test_dicts)

    # Extract target variables
    y_train = df_train['coupon_accepting'].values
    y_test = df_test['coupon_accepting'].values

    # Save datasets as pickle files
    dump_pickle((X_train, y_train), os.path.join("others", "train.pkl"))
    dump_pickle((X_test, y_test), os.path.join("others", "test.pkl"))

    # Log the final preprocessor
    with mlflow.start_run():
        with open("others/final_preprocessor.b", "wb") as f_out:
            pickle.dump(dict_vector, f_out)
        mlflow.log_artifact("others/final_preprocessor.b", artifact_path="final_preprocessor")

@flow
def main_flow(
    lr: float = 0.01,
    mc: float = 1.0,
    md: int = 30,
    rl: float = 0.02,
    ra: float = 0.02,
    prepare_test: bool = False,
) -> None:
    """
    Main training pipeline.

    Args:
        lr (float): Learning rate.
        mc (float): Min child weight.
        md (int): Max depth.
        rl (float): Regularization lambda.
        ra (float): Regularization alpha.
        prepare_test (bool): Whether to prepare the test set.
    """
    # Input data path
    input_path = "https://archive.ics.uci.edu/static/public/603/in+vehicle+coupon+recommendation.zip"
    print("Train path: ", input_path)

    # Set up MLflow
    mlflow.set_tracking_uri(TRACKING_SERVER_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load and preprocess data
    coupon_df = read_data(input_path)

    # Prepare data for training
    X_train, X_val, y_train, y_val, dv, df_full_train, df_test = prepare_data_valid_set(coupon_df)

    # Train the model
    train_best_model(X_train, X_val, y_train, y_val, dv, lr, mc, md, rl, ra)

    # Optionally prepare the test set
    if prepare_test:
        prepare_data_test_set(df_full_train, df_test)

if __name__ == "__main__":
    # Set up argument parser for command-line parameter tuning
    parser = argparse.ArgumentParser(description='Tune hyperparameters for model training.')
    parser.add_argument('-lr', default=0.01, type=float, help='Learning rate')
    parser.add_argument('-md', default=30, type=int, help='Max depth')
    parser.add_argument('-mc', default=1.0, type=float, help='Min child weight')
    parser.add_argument('-rl', default=0.02, type=float, help='Regularization lambda')
    parser.add_argument('-ra', default=0.02, type=float, help='Regularization alpha')
    parser.add_argument('-test', default=False, type=bool, help='Prepare test set')
    args = parser.parse_args()

    # Run the main flow with parsed arguments
    main_flow(args.lr, args.mc, args.md, args.rl, args.ra, args.test)