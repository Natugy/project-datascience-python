#%%
import sys
from pathlib import Path

# Ajouter le root du projet au path
PROJECT_ROOT = Path.cwd().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import numpy as np
import pandas as pd

# Figures
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Existing utility function
from LR_classifiers import afficher_graphiques_metrics

import warnings
warnings.filterwarnings('ignore')

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

from sklearn.inspection import permutation_importance
from sklearn.calibration import CalibrationDisplay

# WandB
import wandb
import joblib

# Data loading (same as used for XGBoost)
from model_xgboost import load_and_prepare_data

# KNN classifier
from sklearn.neighbors import KNeighborsClassifier

# Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV

#%%
# Configuration des chemins
MODELS_DIR = PROJECT_ROOT / "models_saved"
FIGURES_DIR = PROJECT_ROOT / "figures" / "milestone2"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

#%%
# Configure WANDB
WANDB_ENTITY = "IFT6758-2025"
WANDB_PROJECT = "IFT6758-2025"

wandb.login(key="c0237e1d7cc9e5b7d3a6ecf6d37074b81d5298be")

run = wandb.init(
    project=WANDB_PROJECT,
    name="knn-tuning-feature-selection",
    config={"model": "KNN (tuned)"}
)

#%%
# Load and prepare data
X_train, y_train, X_val, y_val, X_test, y_test = load_and_prepare_data()

print(f"Training size: {len(X_train)} samples")
print(f"Validation size: {len(X_val)} samples")

#%%
# 1. Base KNN model
knn = KNeighborsClassifier(
    weights='distance'  # helps with class imbalance
)

#%%
# 2. Hyperparameter search grid
param_grid = {
    "n_neighbors": np.arange(3, 50, 2),
    "weights": ["uniform", "distance"],
    "p": [1, 2],  # Manhattan or Euclidean distance
    "leaf_size": np.arange(20, 50, 5)
}

#%%
# 3. RandomizedSearchCV
search = RandomizedSearchCV(
    estimator=knn,
    param_distributions=param_grid,
    n_iter=60,
    scoring="roc_auc",
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)

best_knn = search.best_estimator_
print("\nBest Hyperparameters:")
print(search.best_params_)

wandb.log({"best_params": search.best_params_})

#%%
# 4. Evaluate tuned KNN
y_scores = best_knn.predict_proba(X_val)[:, 1]  # KNN supports this when using classification

y_pred = (y_scores >= 0.5).astype(int)

metrics = {
    "AUC": roc_auc_score(y_val, y_scores),
    "Accuracy": accuracy_score(y_val, y_pred),
    "Precision": precision_score(y_val, y_pred, zero_division=0),
    "Recall": recall_score(y_val, y_pred, zero_division=0),
    "F1": f1_score(y_val, y_pred, zero_division=0)
}

print("\nValidation Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

wandb.log(metrics)

#%%
# 5. Evaluation plots
fig_knn = afficher_graphiques_metrics(y_val, y_scores, "KNN_Tuned")
wandb.log({"evaluation_plots": wandb.Image(fig_knn)})

#%%
# 6. Permutation importance
print("\nComputing permutation importances...")

perm = permutation_importance(
    best_knn,
    X_val,
    y_val,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

fi_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance": perm.importances_mean
}).sort_values(by="importance", ascending=False)

print(fi_df)

wandb.log({"feature_importances": wandb.Table(dataframe=fi_df)})

plt.figure(figsize=(10, 6))
sns.barplot(data=fi_df, x="importance", y="feature")
plt.title("KNN Permutation Importances")
plt.tight_layout()
wandb.log({"feature_importances_plot": wandb.Image(plt)})
plt.close()

#%%
# 7. Feature Selection (Top-K)
TOP_K = 10
selected_features = fi_df.head(TOP_K)["feature"].tolist()

print("\nSelected Top-K Features:")
print(selected_features)
wandb.log({"selected_features": selected_features})

# Retrain reduced KNN
reduced_knn = best_knn
reduced_knn.fit(X_train[selected_features], y_train)

y_scores_reduced = reduced_knn.predict_proba(X_val[selected_features])[:, 1]
y_pred_reduced = (y_scores_reduced >= 0.5).astype(int)

metrics_reduced = {
    "reduced_AUC": roc_auc_score(y_val, y_scores_reduced),
    "reduced_Accuracy": accuracy_score(y_val, y_pred_reduced),
    "reduced_Precision": precision_score(y_val, y_pred_reduced, zero_division=0),
    "reduced_Recall": recall_score(y_val, y_pred_reduced, zero_division=0),
    "reduced_F1": f1_score(y_val, y_pred_reduced, zero_division=0)
}

print("\nMetrics (Reduced Features):")
for k, v in metrics_reduced.items():
    print(f"{k}: {v:.4f}")

wandb.log(metrics_reduced)

#%%
# 8. Save Model + W&B artifact
model_path = "knn_tuned.pkl"
joblib.dump(best_knn, model_path)

artifact = wandb.Artifact("knn-tuned-model", type="model")
artifact.add_file(model_path)
run.log_artifact(artifact)

run.finish()