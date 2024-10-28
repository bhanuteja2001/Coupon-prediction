"""
Unit tests for the model deployment module.

This module contains tests for the ModelService class in the model_deployment_only_model module.
It focuses on testing the prepare_features and convert_features_to_dmatrix methods.
"""

import pandas as pd
import pytest
import xgboost as xgb

from model_deployment_only_model import ModelService

@pytest.fixture
def model_service():
    """Fixture to create a ModelService instance for tests."""
    return ModelService(model=None, model_version=None)

@pytest.fixture
def test_data():
    """Fixture to provide test data for the prepare_features method."""
    return {
        "destination": ["No Urgent Place", "Home", "Work"],
        "weather": ["Sunny", "Rainy", "Snowy"],
        "time": ["10AM", "10PM", "7AM"],
        "coupon": ["Coffee House", "Coffee House", "Coffee House"],
        "expiration": ["2h", "2h", "1d"],
        "direction_same": [0, 1, 1],
        "direction_opp": [1, 0, 0],
        "Y": [0, 0, 0],
    }

def test_prepare_features(model_service, test_data):
    """
    Test that prepare_features() returns the expected result.

    This test ensures that the prepare_features method correctly transforms
    the input data into the expected format for model prediction.
    """
    actual_result = model_service.prepare_features(test_data)

    expected_result = pd.DataFrame({
        "destination": ["No Urgent Place", "Home", "Work"],
        "weather": ["Sunny", "Rainy", "Snowy"],
        "time": ["10AM", "10PM", "7AM"],
        "coupon": ["Coffee House", "Coffee House", "Coffee House"],
        "expiration": ["2h", "2h", "1d"],
        "same_direction": [0, 1, 1],
        "coupon_accepting": [0, 0, 0],
    })

    pd.testing.assert_frame_equal(actual_result, expected_result)

def test_convert_to_dmatrix(model_service):
    """
    Test that convert_features_to_dmatrix() returns a correct type of the result.

    This test verifies that the convert_features_to_dmatrix method correctly
    transforms a pandas DataFrame into an XGBoost DMatrix object.
    """
    test_df = pd.DataFrame({
        "destination": ["No Urgent Place", "Home", "Work"],
        "weather": ["Sunny", "Rainy", "Snowy"],
        "time": ["10AM", "10PM", "7AM"],
        "coupon": ["Coffee House", "Coffee House", "Coffee House"],
        "expiration": ["2h", "2h", "1d"],
        "same_direction": [0, 1, 1],
        "coupon_accepting": [0, 0, 0],
    })

    try:
        actual_result = model_service.convert_features_to_dmatrix(test_df)
        assert isinstance(actual_result, xgb.DMatrix)
    except FileNotFoundError:
        pytest.skip("Pre-processor file not found. Skipping test.")