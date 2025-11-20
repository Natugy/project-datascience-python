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
    f1_score, roc_auc_score, roc_curve
)

# WandB
import wandb
import joblib

# Data loader
from model_xgboost import load_and_prepare_data

# AdaBoost
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# For hyperparameter tuning
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
    name="adaboost-tuning-and-feature-selection",
    config={"model": "AdaBoost (tuned)"}
)

#%%
# Charger et préparer les données
X_train, y_train, X_val, y_val, X_test, y_test = load_and_prepare_data()

# Compute scale_pos_weight automatically
pos = y_train.sum()
neg = len(y_train) - pos
scale_pos_weight = neg / pos
print(f"scale_pos_weight = {scale_pos_weight:.2f}")

#%%
# 1. Define Base Estimator (Decision Tree)
base_tree = DecisionTreeClassifier(
    random_state=42
)

ada = AdaBoostClassifier(
    estimator=base_tree,
    random_state=42
)

#%%
# 2. Hyperparameter Search Grid
param_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "estimator__max_depth": [1, 2, 3],
    "estimator__min_samples_split": [2, 5, 10],
    "estimator__min_samples_leaf": [1, 2, 4],
    "estimator__class_weight": [
        None,
        "balanced", 
        {"balanced": None}
    ]
}

#%%
# 3. RandomizedSearchCV
search = RandomizedSearchCV(
    estimator=ada,
    param_distributions=param_grid,
    n_iter=100,
    scoring="roc_auc",
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)

best_ada = search.best_estimator_

print("\nBest Hyperparameters:")
print(search.best_params_)

wandb.log({"best_params": search.best_params_})

#%%
# 4. Evaluate tuned AdaBoost
y_scores = best_ada.predict_proba(X_val)[:, 1]
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
# 5. Log evaluation plots
fig_adaboost = afficher_graphiques_metrics(y_val, y_scores, "AdaBoost_Tuned")
wandb.log({"evaluation_plots": wandb.Image(fig_adaboost)})

#%%
# 6. Feature importance  
importances = best_ada.feature_importances_
feature_names = list(X_train.columns)

fi_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print(fi_df)

wandb.log({"feature_importances": wandb.Table(dataframe=fi_df)})

plt.figure(figsize=(10, 6))
sns.barplot(data=fi_df, x="importance", y="feature")
plt.title("AdaBoost Feature Importances")
plt.tight_layout()
wandb.log({"feature_importances_plot": wandb.Image(plt)})
plt.close()

#%%
# integrate XGBoost-style feature selection methods
print("\n===== FEATURE SELECTION (XGBoost METHODS ADAPTED FOR ADABOOST) =====")

TOP_K = 10

# scale once for all selection methods
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)

# ---- Mutual Information ----
features_mi, _ = perform_statistical_feature_selection(
    X_train_scaled, y_train,
    n_features_to_select=TOP_K,
    method="mutual_info",
    output_path=str(FIGURES_DIR / "adaboost_mi.png")
)

# ---- RFE ----
features_rfe, _ = perform_recursive_feature_elimination(
    model_class=AdaBoostClassifier,
    X_train=X_train_scaled, y_train=y_train,
    X_val=X_val_scaled, y_val=y_val,
    n_features_to_select=TOP_K,
    estimator=DecisionTreeClassifier(max_depth=2)
)

# ---- L1 Regularization ----
features_l1, _ = perform_l1_regularization_selection(
    X_train_scaled, y_train, X_val_scaled, y_val,
    n_features_to_select=TOP_K,
    output_path=str(FIGURES_DIR / "adaboost_l1.png")
)

# ---- Sequential Forward Selection ----
features_sfs, _ = perform_sequential_feature_selection(
    model_class=AdaBoostClassifier,
    X_train=X_train_scaled, y_train=y_train,
    X_val=X_val_scaled, y_val=y_val,
    n_features_to_select=TOP_K,
    direction="forward",
    estimator=DecisionTreeClassifier(max_depth=2)
)

# ---- AdaBoost own importances (original) ----
features_importance = fi_df.head(TOP_K)["feature"].tolist()

all_fs = {
    "importance": features_importance,
    "mutual_info": features_mi,
    "rfe": features_rfe,
    "l1_regularization": features_l1,
    "sfs_forward": features_sfs
}

wandb.log({"feature_selection_sets": all_fs})

print("\nSelected features by method:")
for k, v in all_fs.items():
    print(f"{k}: {v}")

#%%
# Train models on each feature subset
def evaluate_subset(label, feat_list):
    best_ada.fit(X_train[feat_list], y_train)
    p = best_ada.predict_proba(X_val[feat_list])[:, 1]
    preds = (p >= 0.5).astype(int)

    m = {
        f"{label}_AUC": roc_auc_score(y_val, p),
        f"{label}_Accuracy": accuracy_score(y_val, preds),
        f"{label}_Precision": precision_score(y_val, preds, zero_division=0),
        f"{label}_Recall": recall_score(y_val, preds, zero_division=0),
        f"{label}_F1": f1_score(y_val, preds, zero_division=0),
    }
    wandb.log(m)
    return m

metrics_all = {}
for key, feats in all_fs.items():
    metrics_all[key] = evaluate_subset(f"AdaBoost_{key}", feats)

best_method = max(metrics_all, key=lambda k: metrics_all[k][f"AdaBoost_{k}_AUC"])
best_feats = all_fs[best_method]

best_ada.fit(X_train[best_feats], y_train)
joblib.dump(best_ada, f"adaboost_best_{best_method}.pkl")

#%%
# 8. Save Model + W&B Artifact
model_path = "adaboost_tuned.pkl"
joblib.dump(best_ada, model_path)

artifact = wandb.Artifact("adaboost-tuned-model", type="model")
artifact.add_file(model_path)
run.log_artifact(artifact)

run.finish()
