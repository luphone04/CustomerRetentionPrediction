#!/usr/bin/env python3
"""
Churn Prediction Pipeline
Trains XGBoost model with Optuna hyperparameter tuning
"""

import os
import sys
import time
import argparse
import json
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, precision_score, recall_score,
    f1_score, roc_auc_score
)
from xgboost import XGBClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.utils.validate_data import validate_telco_data

THRESHOLD = 0.25


def main(args):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # MLflow setup
    if args.mlflow_uri:
        mlruns_path = args.mlflow_uri
    else:
        db_path = os.path.join(project_root, "mlflow.db").replace("\\", "/")
        mlruns_path = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run():
        mlflow.log_param("model", "xgboost")
        mlflow.log_param("threshold", THRESHOLD)
        mlflow.log_param("test_size", args.test_size)

        # Stage 1: Load & Validate
        print("Loading data...")
        df = load_data(args.input)
        print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        print("Validating data...")
        is_valid, failed = validate_telco_data(df)
        mlflow.log_metric("data_quality_pass", int(is_valid))

        if not is_valid:
            mlflow.log_text(json.dumps(failed, indent=2), artifact_file="failed_expectations.json")
            raise ValueError(f"Data validation failed: {failed}")
        print("Validation passed")

        # Stage 2: Preprocess
        print("Preprocessing...")
        df = preprocess_data(df)

        processed_path = os.path.join(project_root, "data", "processed", "telco_churn_processed.csv")
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path, index=False)
        print(f"Saved processed data: {df.shape}")

        # Stage 3: Feature Engineering
        print("Building features...")
        target = args.target
        if target not in df.columns:
            raise ValueError(f"Target '{target}' not found")

        df_enc = build_features(df, target_col=target)

        for c in df_enc.select_dtypes(include=["bool"]).columns:
            df_enc[c] = df_enc[c].astype(int)
        print(f"Features built: {df_enc.shape[1]} columns")

        # Save feature metadata
        artifacts_dir = os.path.join(project_root, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        feature_cols = list(df_enc.drop(columns=[target]).columns)

        with open(os.path.join(artifacts_dir, "feature_columns.json"), "w") as f:
            json.dump(feature_cols, f)
        mlflow.log_text("\n".join(feature_cols), artifact_file="feature_columns.txt")

        preprocessing_artifact = {"feature_columns": feature_cols, "target": target}
        joblib.dump(preprocessing_artifact, os.path.join(artifacts_dir, "preprocessing.pkl"))
        mlflow.log_artifact(os.path.join(artifacts_dir, "preprocessing.pkl"))

        # Stage 4: Train/Test Split
        print("Splitting data...")
        X = df_enc.drop(columns=[target])
        y = df_enc[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=args.test_size,
            stratify=y,
            random_state=42
        )
        print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

        # Class imbalance weight
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        # Stage 5: Optuna Hyperparameter Tuning
        print(f"Tuning XGBoost with Optuna ({args.n_trials} trials)...")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 300, 800),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
                'scale_pos_weight': scale_pos_weight,
                'random_state': 42,
                'n_jobs': -1,
                'eval_metric': 'logloss'
            }

            model = XGBClassifier(**params)
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]
            y_pred = (proba >= THRESHOLD).astype(int)
            return recall_score(y_test, y_pred, pos_label=1)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=args.n_trials)

        print(f"Best Params: {study.best_params}")
        print(f"Best Recall: {study.best_value:.3f}")

        # Log best params to MLflow
        for param_name, param_value in study.best_params.items():
            mlflow.log_param(f"best_{param_name}", param_value)

        # Stage 6: Train Final Model with Best Params
        print("Training final model with best params...")

        best_params = study.best_params
        best_params.update({
            "random_state": 42,
            "n_jobs": -1,
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "logloss"
        })

        model = XGBClassifier(**best_params)

        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        mlflow.log_metric("train_time", train_time)
        print(f"Optimized XGBoost Training Time: {train_time:.2f} seconds")

        # Stage 7: Evaluate
        t1 = time.time()
        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= THRESHOLD).astype(int)
        pred_time = time.time() - t1
        mlflow.log_metric("pred_time", pred_time)
        print(f"Optimized XGBoost Prediction Time: {pred_time:.4f} seconds")

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, proba)

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # Stage 8: Save Model
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(classification_report(y_test, y_pred, digits=3))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Churn prediction pipeline")
    p.add_argument("--input", type=str, required=True, help="Path to CSV")
    p.add_argument("--target", type=str, default="Churn")
    p.add_argument("--n_trials", type=int, default=30)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--experiment", type=str, default="Telco Churn")
    p.add_argument("--mlflow_uri", type=str, default=None)

    args = p.parse_args()
    main(args)
