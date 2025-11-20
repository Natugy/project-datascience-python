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
from sklearn.calibration import CalibrationDisplay

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
    "n_estimators": [200, 300, 400, 500],
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
# 5. Log evaluation plots (ROC, calibration, cumulative gain, goal-rate curve)
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
# 7. Feature Selection  
# Keep top-k important features (example: top 10)
TOP_K = 10
selected_features = fi_df.head(TOP_K)["feature"].tolist()

print("\nSelected Features:")
print(selected_features)

wandb.log({"selected_features": selected_features})

# Train model using only selected features
best_ada_reduced = best_ada
best_ada_reduced.fit(X_train[selected_features], y_train)

y_scores_reduced = best_ada_reduced.predict_proba(X_val[selected_features])[:, 1]
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
# 8. Save Model + W&B Artifact
model_path = "adaboost_tuned.pkl"
joblib.dump(best_ada, model_path)

artifact = wandb.Artifact("adaboost-tuned-model", type="model")
artifact.add_file(model_path)
run.log_artifact(artifact)

run.finish()