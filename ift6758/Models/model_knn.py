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

# Existing utility
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

# Data loading
from model_xgboost import load_and_prepare_data

# KNN classifier
from sklearn.neighbors import KNeighborsClassifier

# Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV

# import feature selection utilities
from ift6758.models.model_xgboost import (
    perform_statistical_feature_selection,
    perform_recursive_feature_elimination,
    perform_l1_regularization_selection,
    perform_sequential_feature_selection
)

from sklearn.preprocessing import StandardScaler

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
    weights='distance'
)

#%%
# 2. Hyperparameter search
param_grid = {
    "n_neighbors": np.arange(3, 50, 2),
    "weights": ["uniform", "distance"],
    "p": [1, 2],
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
# 4. Evaluation
y_scores = best_knn.predict_proba(X_val)[:, 1]
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
# 7. Feature Selection (Top-K from KNN permutation)
TOP_K = 10
selected_perm_features = fi_df.head(TOP_K)["feature"].tolist()

print("\nSelected Top-K Features (Permutation Importance):")
print(selected_perm_features)
wandb.log({"selected_features_perm": selected_perm_features})

#%%
# apply the same feature-selection strategies as XGBoost/AdaBoost
print("\n===== FEATURE SELECTION (XGBoost Methods Adapted for KNN) =====")

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)

# ---- Mutual Information ----
features_mi, _ = perform_statistical_feature_selection(
    X_train_scaled, y_train,
    n_features_to_select=TOP_K,
    method="mutual_info",
    output_path=str(FIGURES_DIR / "knn_mi.png")
)

# ---- RFE ----
features_rfe, _ = perform_recursive_feature_elimination(
    model_class=KNeighborsClassifier,
    X_train=X_train_scaled, y_train=y_train,
    X_val=X_val_scaled, y_val=y_val,
    n_features_to_select=TOP_K,
    n_neighbors=5
)

# ---- L1 Regularization ----
features_l1, _ = perform_l1_regularization_selection(
    X_train_scaled, y_train,
    X_val_scaled, y_val,
    n_features_to_select=TOP_K,
    output_path=str(FIGURES_DIR / "knn_l1.png")
)

# ---- Sequential Forward Selection ----
features_sfs, _ = perform_sequential_feature_selection(
    model_class=KNeighborsClassifier,
    X_train=X_train_scaled, y_train=y_train,
    X_val=X_val_scaled, y_val=y_val,
    n_features_to_select=TOP_K,
    direction="forward",
    n_neighbors=5
)

all_feature_sets = {
    "permutation": selected_perm_features,
    "mutual_info": features_mi,
    "rfe": features_rfe,
    "l1_regularization": features_l1,
    "sfs_forward": features_sfs
}

wandb.log({"feature_sets_knn": all_feature_sets})

print("\nSelected features by method:")
for k, v in all_feature_sets.items():
    print(f"{k}: {v}")

#%%
# Train and evaluate each subset with KNN
def evaluate_subset(label, features):
    best_knn.fit(X_train[features], y_train)
    scores = best_knn.predict_proba(X_val[features])[:, 1]
    preds = (scores >= 0.5).astype(int)

    m = {
        f"{label}_AUC": roc_auc_score(y_val, scores),
        f"{label}_Accuracy": accuracy_score(y_val, preds),
        f"{label}_Precision": precision_score(y_val, preds, zero_division=0),
        f"{label}_Recall": recall_score(y_val, preds, zero_division=0),
        f"{label}_F1": f1_score(y_val, preds, zero_division=0),
    }
    wandb.log(m)
    return m

results = {}
for label, feats in all_feature_sets.items():
    results[label] = evaluate_subset(f"KNN_{label}", feats)

#%%
# 8. Save model + WandB artifact
model_path = "knn_tuned.pkl"
joblib.dump(best_knn, model_path)

artifact = wandb.Artifact("knn-tuned-model", type="model")
artifact.add_file(model_path)
run.log_artifact(artifact)

run.finish()