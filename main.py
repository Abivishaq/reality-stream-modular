# main.py

import os
import pandas as pd
import cudf
import cupy as cp
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from report_prep.report_preparer import ReportPreparer

from model.model_trainer import ModelTrainer


from model.model_trainer_v2 import train_multiple_models  # assuming this exists as per your modularization

from datamanager.data_loader import get_data

# report folder preparation #
report_preparer = ReportPreparer()
report_preparer.setup()

# ---------- Step 1: Prepare CPU Data ---------- #
# from datamanager import load_full_dataset  # hypothetical helper
X_total_cpu, y_total_cpu = get_data()

X_train_pd, X_val_pd, y_train_np, y_val_np = train_test_split(
    X_total_cpu.fillna(0),
    y_total_cpu,
    test_size=0.2,
    stratify=y_total_cpu,
    random_state=42
)

# ---------- Step 2: Filter Numeric Columns ---------- #
X_train_pd = X_train_pd.select_dtypes(include=[np.number])
X_val_pd = X_val_pd.select_dtypes(include=[np.number])

# ---------- Step 3: Apply SMOTE ---------- #
smote = SMOTE(random_state=42)
X_train_smote_pd, y_train_smote_np = smote.fit_resample(X_train_pd, y_train_np)

# ---------- Step 4: Convert to GPU ---------- #
X_train_smote = cudf.DataFrame.from_pandas(X_train_smote_pd)
y_train_smote = cp.asarray(y_train_smote_np)
X_val = cudf.DataFrame.from_pandas(X_val_pd)
y_val = cp.asarray(y_val_np)

print(f"After SMOTE: X_train_smote {X_train_smote.shape}, X_val {X_val.shape}")

# ---------- Step 5: Train Models ---------- #
results_smote = train_multiple_models(
    X_train=X_train_smote,
    y_train=y_train_smote,
    X_test=X_val,
    y_test=y_val,
    model_types=['rfc', 'xgboost', 'lr', 'mlp', 'svm'],
    random_state=42
)

# ---------- Step 6: Save Report ---------- #
results_smote_df = pd.DataFrame([{
    "Model": r["model_type"],
    "Accuracy": r["accuracy"],
    "ROC_AUC": r["roc_auc"],
    "F1_Score": r["f1_score"],
    "Precision": r["precision"],
    "Recall": r["recall"],
    "GMean": r["gmean"],
    "Training_Time_Seconds": r["time"]
} for r in results_smote])

REPORT_FOLDER = "report" 

os.makedirs(REPORT_FOLDER, exist_ok=True)
results_smote_df.to_csv(os.path.join(REPORT_FOLDER, "model_performance_report_smote.csv"), index=False)
print("Report saved to model_performance_report_smote.csv")
