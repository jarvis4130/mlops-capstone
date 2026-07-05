import pandas as pd
import glob
import pickle
import os
from sklearn.preprocessing import OrdinalEncoder

RAW_DIR = "data/raw"
MODEL_DIR = "models"

FEATURES = ["Area", "Location", "No. of Bedrooms", "Resale", "MaintenanceStaff",
            "Gymnasium", "SwimmingPool", "LiftAvailable", "CarParking", "City"]
TARGET = "Price"

def load_data():
    files = glob.glob(f"{RAW_DIR}/*.csv")
    dfs = []
    for f in files:
        if "sample" in f:
            continue
        city = os.path.basename(f).replace(".csv", "")
        tmp = pd.read_csv(f)
        tmp["City"] = city
        dfs.append(tmp)
    df = pd.concat(dfs, ignore_index=True)
    return df[FEATURES + [TARGET]].dropna()

def remove_outliers(df):
    q99_price = df[TARGET].quantile(0.99)
    q99_area = df["Area"].quantile(0.99)
    return df[(df[TARGET] <= q99_price) & (df["Area"] <= q99_area)]

def preprocess(df):
    os.makedirs(MODEL_DIR, exist_ok=True)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df = df.copy()
    df[["Location", "City"]] = enc.fit_transform(df[["Location", "City"]])
    pickle.dump(enc, open(f"{MODEL_DIR}/encoder.pkl", "wb"))
    return df

def load_and_prepare():
    df = load_data()
    df = remove_outliers(df)
    df = preprocess(df)
    return df
