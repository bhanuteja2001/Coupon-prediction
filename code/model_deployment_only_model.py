"""
Model Service Module

This module contains the ModelService class, which provides functionality for
loading, preparing, and using a machine learning model for coupon acceptance prediction.
"""

import pickle
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import xgboost as xgb

class ModelService:
    """
    A service class for managing the coupon acceptance prediction model.

    This class provides methods for loading data, preparing features,
    and making predictions using an XGBoost model.
    """

    def __init__(self, model: Any, model_version: str = None) -> None:
        """
        Initialize the ModelService.

        Args:
            model: The XGBoost model object.
            model_version: The version of the model (optional).
        """
        self.model = model
        self.model_version = model_version

    def load_pickle(self, filename: str) -> Any:
        """
        Load a pickle file from the same directory as this script.

        Args:
            filename: Name of the pickle file to load.

        Returns:
            The unpickled object.
        """
        with open(Path(__file__).parent / filename, "rb") as f_in:
            return pickle.load(f_in)

    def prepare_features(self, coupon_rec: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare features for prediction from raw coupon recommendation data.

        Args:
            coupon_rec: A dictionary containing coupon recommendation data.

        Returns:
            A pandas DataFrame with prepared features.
        """
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

    def convert_features_to_dmatrix(self, dataframe: pd.DataFrame) -> xgb.DMatrix:
        """
        Convert a pandas DataFrame to an XGBoost DMatrix.

        Args:
            dataframe: A pandas DataFrame containing the features.

        Returns:
            An XGBoost DMatrix object.
        """
        features = [
            'destination', 'weather', 'time', 'coupon', 'expiration', 'same_direction'
        ]

        dict_vector = self.load_pickle('others/final_preprocessor.b')

        temp_dicts = dataframe[features].to_dict(orient="records")
        tf_temp_dicts = dict_vector.transform(temp_dicts)
        target = dataframe['coupon_accepting'].values

        return xgb.DMatrix(tf_temp_dicts, label=target)

    def predict(self, features: xgb.DMatrix) -> str:
        """
        Predict coupon acceptance.

        Args:
            features: An XGBoost DMatrix containing the features.

        Returns:
            A string indicating 'Yes' if the coupon is likely to be accepted, 'No' otherwise.
        """
        print(f"Input type: {type(features)}")
        
        # Note: This is a placeholder implementation. In a real scenario,
        # you would use self.model.predict(features) instead of a constant value.
        preds = 0.9
        
        return 'Yes' if preds > 0.5 else 'No'