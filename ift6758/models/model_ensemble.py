#%%
import sys
from pathlib import Path

# Ajouter le root du projet
PROJECT_ROOT = Path.cwd().parent.parent # project-datascience-python
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

# Data loader
from model_xgboost import load_and_prepare_data

# Ensemble model
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV

# Feature-selection utilities (from XGBoost module)
from ift6758.models.model_xgboost import (
    perform_statistical_feature_selection,
    perform_recursive_feature_elimination,
    perform_l1_regularization_selection,
    perform_sequential_feature_selection
)

from sklearn.preprocessing import StandardScaler

#%%
# Paths
MODELS_DIR = PROJECT_ROOT / "ift6758" / "models_saved"
FIGURES_DIR = PROJECT_ROOT / "figures" / "milestone2"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

#%%
# W&B config
WANDB_ENTITY = "IFT6758-2025"
WANDB_PROJECT = "IFT6758-2025"

wandb.login(key="c0237e1d7cc9e5b7d3a6ecf6d37074b81d5298be")

run = wandb.init(
    project=WANDB_PROJECT,
    name="ensemble-voting-feature-selection",
    config={"model": "Ensemble Voting (soft)"}
)

#%%
# Load data
X_train, y_train, X_val, y_val, X_test, y_test = load_and_prepare_data()

print(f"Training size: {len(X_train)}")
print(f"Validation size: {len(X_val)}")


#%%
# 1. Load tuned base learners from previously trained models
# (Make sure the .pkl files exist in your /models_saved directory)
model_lr = joblib.load(MODELS_DIR / "logreg_distance_angle.pkl") # initial Logistic Regression
model_rf = joblib.load(MODELS_DIR / "rf_tuned.pkl")           # Best RF model 
model_ada = joblib.load(MODELS_DIR / "adaboost_tuned.pkl")    # Best AdaBoost model
model_xgb = joblib.load(MODELS_DIR /"xgboost_tuned_fs.pkl")       # Best XGBoost model 
model_knn = joblib.load(MODELS_DIR /"knn_tuned.pkl")            # Best KNN model

# Create a soft-voting ensemble using tuned models
ensemble = VotingClassifier(
    estimators=[
        ("lr", model_lr),
        ("rf", model_rf),
        ("ada", model_ada),
        ("xgb", model_xgb),
        ("knn", model_knn)
    ],
    voting="soft",       # soft = based on predicted probabilities
    weights=[1, 2, 2, 3, 1],   # higher weight for more performant models (e.g., XGBoost, RF)
    n_jobs=-1
)

#%%
# 2. Hyperparameter search grid (on ensemble internal RF + XGB)
param_grid = {
    "rf__n_estimators": [200, 300, 400],
    "rf__max_depth": [10, 20, None],
    "xgb__max_depth": [3, 5, 7],
    "xgb__learning_rate": [0.01, 0.05, 0.1],
    "xgb__n_estimators": [200, 300],
}

#%%
# 3. RandomizedSearchCV
search = RandomizedSearchCV(
    estimator=ensemble,
    param_distributions=param_grid,
    n_iter=30,
    scoring="roc_auc",
    cv=5,
    random_state=42,
    verbose=1,
    n_jobs=-1
)

search.fit(X_train, y_train)
best_ensemble = search.best_estimator_

print("\nBest Hyperparameters:")
print(search.best_params_)

wandb.log({"best_params": search.best_params_})

#%%
# 4. Evaluate tuned ensemble
y_scores = best_ensemble.predict_proba(X_val)[:, 1]
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
fig_ens = afficher_graphiques_metrics(y_val, y_scores, "Ensemble_Tuned")
wandb.log({"evaluation_plots": wandb.Image(fig_ens)})

#%%
# 6. Permutation importance (ensemble weighted effect)
print("\nComputing permutation importances...")

perm = permutation_importance(
    best_ensemble,
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
sns.barplot(data=fi_df.head(20), x="importance", y="feature")
plt.title("Ensemble Permutation Importances")
plt.tight_layout()
wandb.log({"feature_importances_plot": wandb.Image(plt)})
plt.close()

#%%
# 7. Feature Selection (Top 10)
TOP_K = 10
selected_features = fi_df.head(TOP_K)["feature"].tolist()

print("\nSelected Top-K Features:")
print(selected_features)
wandb.log({"selected_features": selected_features})

# Reduced model training
reduced_model = best_ensemble
reduced_model.fit(X_train[selected_features], y_train)

y_scores_reduced = reduced_model.predict_proba(X_val[selected_features])[:, 1]
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
# 8. Advanced Feature-Selection Methods
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)

# ---- Mutual Information ----
features_mi, _ = perform_statistical_feature_selection(
    X_train_scaled, y_train,
    n_features_to_select=TOP_K,
    method="mutual_info",
    output_path=str(FIGURES_DIR / "ensemble_mi.png")
)

# ---- RFE ----
features_rfe, _ = perform_recursive_feature_elimination(
    model_class=RandomForestClassifier,
    X_train=X_train_scaled, y_train=y_train,
    X_val=X_val_scaled, y_val=y_val,
    n_features_to_select=TOP_K,
    n_estimators=200, n_jobs=-1
)

# ---- L1 ----
features_l1, _ = perform_l1_regularization_selection(
    X_train_scaled, y_train,
    X_val_scaled, y_val,
    n_features_to_select=TOP_K,
    output_path=str(FIGURES_DIR / "ensemble_l1.png")
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
    "ensemble_topK_perm": selected_features,
    "mutual_info": features_mi,
    "rfe": features_rfe,
    "l1_regularization": features_l1,
    "sfs_forward": features_sfs
}

wandb.log({"feature_sets_ensemble": all_feature_sets})

print("\nAll feature-selection results:")
for name, feats in all_feature_sets.items():
    print(f"{name}: {feats}")

#%%
# 9. Train on each feature subset
def evaluate_subset(label, feats):
    best_ensemble.fit(X_train[feats], y_train)
    scores = best_ensemble.predict_proba(X_val[feats])[:, 1]
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

results = {label: evaluate_subset(f"Ensemble_{label}", feats)
           for label, feats in all_feature_sets.items()}

#%%
# 10. Save Ensemble Model
model_path = "ensemble_tuned.pkl"
joblib.dump(best_ensemble, model_path)

artifact = wandb.Artifact("ensemble-tuned-model", type="model")
artifact.add_file(model_path)
run.log_artifact(artifact)

run.finish()
# %%
