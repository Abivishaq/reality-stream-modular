# ------------------ Imports ------------------ #
import os
import pandas as pd
import numpy as np
import cudf
import cupy as cp
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from cuml.ensemble import RandomForestClassifier as cuRF
from cuml.linear_model import LogisticRegression as cuLR
from cuml.svm import SVC as cuSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from xgboost import XGBClassifier
import time

# ------------------ Helper Functions ------------------ #
def safe_to_cpu(arr):
    if isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    elif isinstance(arr, (cudf.Series, cudf.DataFrame)):
        return arr.to_numpy()
    else:
        return arr

# ------------------ Training Function ------------------ #
def train_multiple_models(X_train, y_train, X_test, y_test, model_types, random_state=None, n_iter=20):
    if isinstance(X_train, pd.DataFrame):
        X_train = cudf.DataFrame.from_pandas(X_train)
    if isinstance(X_test, pd.DataFrame):
        X_test = cudf.DataFrame.from_pandas(X_test)
    if isinstance(y_train, (np.ndarray, pd.Series)):
        y_train = cp.asarray(y_train)
    if isinstance(y_test, (np.ndarray, pd.Series)):
        y_test = cp.asarray(y_test)

    results = []
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    param_grids = {
        "xgboost": {
            "n_estimators": np.random.randint(20, 50, n_iter).tolist(),
            "learning_rate": np.random.uniform(0.01, 0.1, n_iter).tolist(),
            "max_depth": np.random.randint(2, 4, n_iter).tolist(),
            "min_child_weight": np.random.randint(5, 10, n_iter).tolist(),
            "subsample": np.random.uniform(0.5, 0.7, n_iter).tolist(),
            "colsample_bytree": np.random.uniform(0.5, 0.7, n_iter).tolist(),
            "gamma": np.random.uniform(0.1, 0.5, n_iter).tolist(),
            "reg_alpha": np.random.uniform(0.5, 1.5, n_iter).tolist(),
            "reg_lambda": np.random.uniform(1.0, 3.0, n_iter).tolist(),
        },
        "mlp": {
            "hidden_layer_sizes": [(50,), (100,), (50, 50)],
            "activation": ["relu", "tanh"],
            "solver": ["adam", "sgd"],
            "alpha": np.logspace(-4, -1, n_iter).tolist(),
            "learning_rate_init": np.random.uniform(0.0001, 0.01, n_iter).tolist(),
            "max_iter": [300, 500]
        }
    }

    for model_type in model_types:
        if model_type == "rfc":
            model = cuRF(
                n_estimators=100,
                max_depth=7,
                max_features=0.7,
                random_state=random_state,
                n_streams=1
            )
        elif model_type == "xgboost":
            model = XGBClassifier(
                tree_method="gpu_hist",
                predictor="gpu_predictor",
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=random_state
            )
        elif model_type == "lr":
            model = cuLR(max_iter=1000)
        elif model_type == "svm":
            model = cuSVC(probability=True, kernel='rbf', C=10.0, gamma='auto')
        elif model_type == "mlp":
            model = MLPClassifier(random_state=random_state)
        else:
            print(f"Skipping unsupported model type: {model_type}")
            continue

        print(f"\nTraining model: {model_type.upper()}...")
        start = time.time()

        if model_type in ["xgboost", "mlp"]:
            X_train_pd = X_train.to_pandas()
            X_test_pd = X_test.to_pandas()
            y_train_np = cp.asnumpy(y_train)

            rand_search = RandomizedSearchCV(
                model,
                param_distributions=param_grids[model_type],
                n_iter=n_iter,
                cv=3,
                scoring="accuracy",
                n_jobs=-1,
                verbose=1,
                random_state=random_state
            )
            rand_search.fit(X_train_pd, y_train_np)
            best_model = rand_search.best_estimator_

            y_pred = best_model.predict(X_test_pd)
            y_pred_prob = best_model.predict_proba(X_test_pd)
        else:
            model.fit(X_train, y_train)
            best_model = model
            y_pred = model.predict(X_test)
            y_pred_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        end = time.time()

        y_test_cpu = safe_to_cpu(y_test)
        y_pred_cpu = safe_to_cpu(y_pred)
        y_pred_prob_cpu = safe_to_cpu(y_pred_prob) if y_pred_prob is not None else None

        report = classification_report(y_test_cpu, y_pred_cpu, output_dict=True, zero_division=0)
        accuracy_num = accuracy_score(y_test_cpu, y_pred_cpu)
        roc_auc = roc_auc_score(y_test_cpu, y_pred_prob_cpu[:, 1]) if y_pred_prob_cpu is not None else 0.0
        gmeans_num = (report["0"]["recall"] * report["1"]["recall"]) ** 0.5 if "0" in report and "1" in report else 0.0

        precision = report["1"]["precision"] if "1" in report else 0.0
        recall = report["1"]["recall"] if "1" in report else 0.0
        f1 = report["1"]["f1-score"] if "1" in report else 0.0

        results.append({
            "model_type": model_type,
            "best_model": best_model,
            "accuracy": round(accuracy_num, 4),
            "roc_auc": round(roc_auc, 4),
            "gmean": round(gmeans_num, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "time": round(end - start, 2),
            "classification_report": report,
        })

        print(f"Accuracy: {accuracy_num:.4f}, ROC-AUC: {roc_auc:.4f}, F1-Score: {f1:.4f}")

    return results
