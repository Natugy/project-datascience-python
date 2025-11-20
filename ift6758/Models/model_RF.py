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

# Data loading
from model_xgboost import load_and_prepare_data

# RandomForest classifier
from sklearn.ensemble import RandomForestClassifier

# Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV

# Feature-selection utilities from the XGBoost module
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
    name="rf-tuning-feature-selection",
    config={"model": "RandomForest (tuned)"}
)

#%%
# Load and prepare data
X_train, y_train, X_val, y_val, X_test, y_test = load_and_prepare_data()
print(f"Training size: {len(X_train)}")
print(f"Validation size: {len(X_val)}")

#%%
# 1. Base RandomForest model
rf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

#%%
# 2. Hyperparameter search grid
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30, 40],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False]
}

#%%
# 3. RandomizedSearchCV
search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=70,
    scoring="roc_auc",
    cv=5,
    random_state=42,
    verbose=1,
    n_jobs=-1
)

search.fit(X_train, y_train)
best_rf = search.best_estimator_

print("\nBest Hyperparameters:")
print(search.best_params_)

wandb.log({"best_params": search.best_params_})

#%%
# 4. Evaluate tuned RF
y_scores = best_rf.predict_proba(X_val)[:, 1]
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
fig_rf = afficher_graphiques_metrics(y_val, y_scores, "RF_Tuned")
wandb.log({"evaluation_plots": wandb.Image(fig_rf)})

#%%
# 6. Feature importance (RF Gini)
importances = best_rf.feature_importances_
feature_names = X_train.columns

fi_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

print(fi_df)

wandb.log({"feature_importances": wandb.Table(dataframe=fi_df)})

plt.figure(figsize=(10, 6))
sns.barplot(data=fi_df.head(20), x="importance", y="feature")
plt.title("RandomForest Feature Importances")
plt.tight_layout()
wandb.log({"feature_importances_plot": wandb.Image(plt)})
plt.close()

#%%
# 7. Feature Selection — TOP-K RF importances
TOP_K = 10
selected_features_rf = fi_df.head(TOP_K)["feature"].tolist()

print("\nTop-K Important Features (RF):")
print(selected_features_rf)
wandb.log({"rf_selected_features": selected_features_rf})

# Train reduced model
best_rf_reduced = best_rf
best_rf_reduced.fit(X_train[selected_features_rf], y_train)

y_scores_reduced = best_rf_reduced.predict_proba(X_val[selected_features_rf])[:, 1]
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
# Additional Feature Selection Strategies 
print("\n===== FEATURE SELECTION (advanced methods) =====")

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)

# ---- Mutual Information ----
features_mi, _ = perform_statistical_feature_selection(
    X_train_scaled, y_train,
    n_features_to_select=TOP_K,
    method="mutual_info",
    output_path=str(FIGURES_DIR / "rf_mi.png")
)

# ---- RFE ----
features_rfe, _ = perform_recursive_feature_elimination(
    model_class=RandomForestClassifier,
    X_train=X_train_scaled, y_train=y_train,
    X_val=X_val_scaled, y_val=y_val,
    n_features_to_select=TOP_K,
    n_estimators=200,
    n_jobs=-1
)

# ---- L1 Regularization ----
features_l1, _ = perform_l1_regularization_selection(
    X_train_scaled, y_train,
    X_val_scaled, y_val,
    n_features_to_select=TOP_K,
    output_path=str(FIGURES_DIR / "rf_l1.png")
)

# ---- Sequential Forward Selection ----
features_sfs, _ = perform_sequential_feature_selection(
    model_class=RandomForestClassifier,
    X_train=X_train_scaled, y_train=y_train,
    X_val=X_val_scaled, y_val=y_val,
    n_features_to_select=TOP_K,
    direction="forward",
    n_estimators=200,
    n_jobs=-1
)

all_feature_sets = {
    "rf_importance": selected_features_rf,
    "mutual_info": features_mi,
    "rfe": features_rfe,
    "l1_regularization": features_l1,
    "sfs_forward": features_sfs
}

wandb.log({"feature_sets_rf": all_feature_sets})

print("\nAll feature-selection methods:")
for name, feats in all_feature_sets.items():
    print(f"{name}: {feats}")

#%%
# 9. Train & evaluate RF on each feature subset
def evaluate_subset(label, feats):
    best_rf.fit(X_train[feats], y_train)
    scores = best_rf.predict_proba(X_val[feats])[:, 1]
    preds = (scores >= 0.5).astype(int)

    m = {
        f"{label}_AUC": roc_auc_score(y_val, scores),
        f"{label}_Accuracy": accuracy_score(y_val, preds),
        f"{label}_Precision": precision_score(y_val, preds, zero_division=0),
        f"{label}_Recall": recall_score(y_val, preds, zero_division=0),
        f"{label}_F1": f1_score(y_val, preds, zero_division=0)
    }
    wandb.log(m)
    return m

results = {label: evaluate_subset(f"RF_{label}", feats)
           for label, feats in all_feature_sets.items()}

#%%
# 10. Identify and log the best feature selection method based on AUC
print("\n===== Selecting Best Feature Subset (based on AUC) =====")

# Find best-performing method
best_method = max(results, key=lambda k: results[k][f"RF_{k}_AUC"])
best_feats = all_feature_sets[best_method]

print(f"Best feature selection method: {best_method}")
print(f"Selected features ({len(best_feats)}): {best_feats}")

# Retrain model using best subset
best_rf.fit(X_train[best_feats], y_train)

# Evaluate on validation data
scores_best = best_rf.predict_proba(X_val[best_feats])[:, 1]
preds_best = (scores_best >= 0.5).astype(int)

best_metrics = {
    "best_method": best_method,
    "best_AUC": roc_auc_score(y_val, scores_best),
    "best_Accuracy": accuracy_score(y_val, preds_best),
    "best_Precision": precision_score(y_val, preds_best, zero_division=0),
    "best_Recall": recall_score(y_val, preds_best, zero_division=0),
    "best_F1": f1_score(y_val, preds_best, zero_division=0),
}

print("\nBest Method Validation Metrics:")
for k, v in best_metrics.items():
    if isinstance(v, (int, float)):
        print(f"{k}: {v:.4f}")
    else:
        print(f"{k}: {v}")

# Log results to W&B
wandb.log(best_metrics)
wandb.log({"best_selected_features": best_feats})

# Save final best model
best_model_path = f"rf_best_{best_method}.pkl"
joblib.dump(best_rf, best_model_path)

artifact_best = wandb.Artifact(f"rf-best-{best_method}", type="model")
artifact_best.add_file(best_model_path)
run.log_artifact(artifact_best)

print(f"\nBest model saved as: {best_model_path}")

#%%
# 10. Save Model + WandB artifact
model_path = "rf_tuned.pkl"
joblib.dump(best_rf, model_path)

artifact = wandb.Artifact("rf-tuned-model", type="model")
artifact.add_file(model_path)
run.log_artifact(artifact)

run.finish()