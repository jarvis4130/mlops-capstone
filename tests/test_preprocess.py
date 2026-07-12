import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from preprocess import remove_outliers, FEATURES, TARGET

def make_mock_df(n=100):
    np.random.seed(42)
    df = pd.DataFrame({
        'Area': np.random.randint(500, 5000, n),
        'Location': ['Whitefield'] * n,
        'No. of Bedrooms': np.random.randint(1, 5, n),
        'Resale': np.random.randint(0, 2, n),
        'MaintenanceStaff': np.random.randint(0, 2, n),
        'Gymnasium': np.random.randint(0, 2, n),
        'SwimmingPool': np.random.randint(0, 2, n),
        'LiftAvailable': np.random.randint(0, 2, n),
        'CarParking': np.random.randint(0, 2, n),
        'City': ['Bangalore'] * n,
        'Price': np.random.randint(1_000_000, 10_000_000, n),
    })
    return df

def test_mock_df_shape():
    df = make_mock_df()
    assert df.shape[0] == 100
    assert df.shape[1] == len(FEATURES) + 1

def test_no_nulls():
    df = make_mock_df()
    assert df.isnull().sum().sum() == 0

def test_remove_outliers_reduces_rows():
    df = make_mock_df()
    # Add extreme outliers
    df.loc[0, 'Price'] = 999_999_999_999
    df.loc[1, 'Area'] = 999_999_999
    df_clean = remove_outliers(df)
    assert df_clean.shape[0] < df.shape[0]

def test_target_column_exists():
    df = make_mock_df()
    assert TARGET in df.columns

def test_all_features_exist():
    df = make_mock_df()
    for f in FEATURES:
        assert f in df.columns
