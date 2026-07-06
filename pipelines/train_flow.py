import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from prefect import flow, task

from preprocess import load_and_prepare, FEATURES, TARGET

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

@task
def load_data():
    df = load_and_prepare()
    print(f"Data shape: {df.shape}")
    return df

@task
def split_data(df):
    X = df[FEATURES]
    y = np.log1p(df[TARGET])
    return train_test_split(X, y, test_size=0.2, random_state=42)

@task
def train_linear(X_train, X_test, y_train, y_test):
    mlflow.set_experiment("house-price-prediction")
    with mlflow.start_run(run_name="LinearRegression"):
        lr = LinearRegression().fit(X_train, y_train)
        rmse = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(lr.predict(X_test))))
        mlflow.log_param("model", "LinearRegression")
        mlflow.log_metric("rmse", rmse)
        print(f"Linear Regression RMSE: {rmse:,.0f}")
    return rmse

@task
def train_xgboost(X_train, X_test, y_train, y_test):
    params = {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1, "random_state": 42}
    mlflow.set_experiment("house-price-prediction")
    with mlflow.start_run(run_name="XGBoost"):
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)

        preds = np.expm1(model.predict(X_test))
        rmse = np.sqrt(mean_squared_error(np.expm1(y_test), preds))
        r2 = r2_score(np.expm1(y_test), preds)

        mlflow.log_params(params)
        mlflow.log_metrics({"rmse": rmse, "r2": r2})
        mlflow.xgboost.log_model(model, "model")

        print(f"XGBoost RMSE: {rmse:,.0f}")
        print(f"XGBoost R2:   {r2:.4f}")

        model.save_model(f"{MODEL_DIR}/model.json")
        print("Model saved to models/model.json")
    return rmse

@flow(name="house-price-training-pipeline")
def training_pipeline():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    train_linear(X_train, X_test, y_train, y_test)
    train_xgboost(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    training_pipeline()
