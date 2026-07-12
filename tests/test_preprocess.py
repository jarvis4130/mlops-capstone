import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from preprocess import load_data, remove_outliers, load_and_prepare, FEATURES, TARGET

def test_load_data_shape():
    df = load_data()
    assert df.shape[0] > 0, "DataFrame should have rows"
    assert df.shape[1] == len(FEATURES) + 1, "DataFrame should have correct number of columns"

def test_no_nulls():
    df = load_data()
    assert df.isnull().sum().sum() == 0, "Data should have no nulls"

def test_remove_outliers_reduces_rows():
    df = load_data()
    df_clean = remove_outliers(df)
    assert df_clean.shape[0] < df.shape[0], "Outlier removal should reduce rows"

def test_target_column_exists():
    df = load_data()
    assert TARGET in df.columns, f"{TARGET} column should exist"

def test_all_features_exist():
    df = load_data()
    for f in FEATURES:
        assert f in df.columns, f"Feature {f} should exist in dataframe"
