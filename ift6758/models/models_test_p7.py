"""
Script: evaluate_registered_models.py
Purpose: Fetch 'best' models from W&B artifacts, evaluate them on the 2020–2021
         regular season and playoff datasets (split from test_data.csv),
         compute metrics, and generate two separate 4×4 figures.
"""

# %%
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve
)
from sklearn.calibration import CalibrationDisplay

# === Project structure ===
current_file = Path(__file__).absolute()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

# === W&B setup ===
wandb.login(key=os.environ.get("WANDB_API_KEY"))
ENTITY = "qi-li-1-universit-de-montr-al"
PROJECT = "IFT6758-2025"

# -------------------------------------------------------------------------
# Helper: Load model artifact
# -------------------------------------------------------------------------
def load_model_from_artifact(artifact_name: str, alias: str = "best"):
    artifact_path = f"{ENTITY}/{PROJECT}/{artifact_name}:{alias}"
    print(f"Downloading model artifact: {artifact_path}")
    artifact = wandb.use_artifact(artifact_path, type="model")
    model_dir = artifact.download()
    print(f"Artifact version: {artifact.version} | Name: {artifact.name}")

    for file in Path(model_dir).glob("*.pkl"):
        print(f"  Loaded model file: {file.name}")
        return joblib.load(file)

    raise FileNotFoundError(f"No .pkl file found in {artifact_path}")

# -------------------------------------------------------------------------
# Helper: Auto-align features
# -------------------------------------------------------------------------
def align_features(model, X):
    expected = getattr(model, "feature_names_in_", X.columns)

    for col in expected:
        if col not in X.columns:
            X[col] = 0

    return X[expected]

# -------------------------------------------------------------------------
# Helper: Evaluate model
# -------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test):
    X_test = align_features(model, X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    metrics = {
        "AUC": roc_auc_score(y_test, y_pred_proba),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
    }
    return y_pred_proba, metrics

# -------------------------------------------------------------------------
# Helper: Plot 4 standard graphs for a given dataset
# -------------------------------------------------------------------------
def plot_dataset_figures(y_scores_dict, y_true_dict, model_colors, title, results):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    fig.suptitle(title, fontsize=16, fontweight="bold")

    # =========== ROC CURVE ===========
    for model_name, scores in y_scores_dict.items():

        is_playoff_plot = "Playoff" in title or "playoff" in title.lower()

        if is_playoff_plot:
            auc_val = results[model_name + " (Playoff)"]["AUC"]
        else:
            auc_val = results[model_name]["AUC"]

        fpr, tpr, _ = roc_curve(y_true_dict[model_name], scores)

        axes[0].plot(
            fpr, tpr,
            color=model_colors[model_name],
            label=f"{model_name} (AUC={auc_val:.3f})"
        )

    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].set_title("ROC Curve")
    axes[0].legend()

    # ========= GOAL RATE PER PERCENTILE =========
    n_bins = 10
    for model_name, y_scores in y_scores_dict.items():
        y_ref = y_true_dict[model_name]
        idx = np.argsort(y_scores)[::-1]
        sorted_labels = np.array(y_ref)[idx]

        bin_size = len(sorted_labels) // n_bins
        centiles, rates = [], []

        for i in range(n_bins):
            start, end = i * bin_size, (i + 1) * bin_size
            rates.append(np.mean(sorted_labels[start:end]))
            centiles.append(100 - (i + 0.5) * (100 / n_bins))

        axes[1].plot(centiles, rates, marker="o",
                     label=model_name,
                     color=model_colors[model_name])

    axes[1].invert_xaxis()
    axes[1].set_title("Goal Rate per Percentile")
    axes[1].legend()

    # ========= CUMULATIVE GOAL PROPORTION =========
    for model_name, y_scores in y_scores_dict.items():
        y_ref = y_true_dict[model_name]
        idx = np.argsort(y_scores)[::-1]
        sorted_labels = np.array(y_ref)[idx]

        total_goals = np.sum(sorted_labels == 1)
        cum_goals = np.cumsum(sorted_labels == 1) / total_goals
        centiles = np.arange(1, len(y_scores) + 1) / len(y_scores) * 100

        axes[2].plot(
            centiles, cum_goals,
            label=model_name,
            color=model_colors[model_name]
        )

    axes[2].set_title("Cumulative Goal Proportion")
    axes[2].legend()

    # ========= CALIBRATION CURVE =========
    for model_name, y_scores in y_scores_dict.items():
        y_ref = y_true_dict[model_name]

        CalibrationDisplay.from_predictions(
            y_ref, y_scores, n_bins=10,
            ax=axes[3],
            name=model_name
        )

    axes[3].plot([0, 1], [0, 1], "k--")
    axes[3].set_title("Calibration Curve")

    plt.tight_layout()
    return fig

# =============================================================================
# MAIN SCRIPT
# =============================================================================
def main():

    wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name="evaluate-best-artifacts",
        job_type="evaluation",
    )

    test_path = project_root / "data" / "processed" / "test_data.csv"
    df_test = pd.read_csv(test_path)

    df_test["idGame"] = df_test["idGame"].astype(str)
    playoffs_mask = df_test["idGame"].str[4:6] == "03"

    df_regular = df_test[~playoffs_mask].copy()
    df_playoffs = df_test[playoffs_mask].copy()

    X_reg = df_regular.drop(columns=["is_goal"])
    y_reg = df_regular["is_goal"]

    X_po = df_playoffs.drop(columns=["is_goal"])
    y_po = df_playoffs["is_goal"]

    models_registry = {
        "LR (distance)": "logreg-distance",
        "LR (angle)": "logreg-angle",
        "LR (distance+angle)": "logreg-distance-angle",
        "XGBoost (tuned+FS)": "xgboost_tuned_fs",
        "Ensemble (final)": "ensemble-tuned-model",
    }

    model_colors = {
        name: color for name, color in zip(
            models_registry.keys(),
            ["blue", "red", "green", "purple", "orange"]
        )
    }

    # === FIX: create results dict ===
    results = {}

    y_scores_reg = {}
    y_scores_po = {}
    y_true_reg = {}
    y_true_po = {}

    # =====================================================
    # Evaluate all models
    # =====================================================
    for model_name, artifact_name in models_registry.items():
        try:
            model = load_model_from_artifact(artifact_name)

            scores_reg, metrics_reg = evaluate_model(model, X_reg.copy(), y_reg)
            y_scores_reg[model_name] = scores_reg
            y_true_reg[model_name] = y_reg.values
            results[model_name] = metrics_reg

            scores_po, metrics_po = evaluate_model(model, X_po.copy(), y_po)
            y_scores_po[model_name] = scores_po
            y_true_po[model_name] = y_po.values
            results[model_name + " (Playoff)"] = metrics_po

        except Exception as e:
            print(f"⚠️ Error evaluating {model_name}: {e}")

    # =====================================================
    # Generate and log figures
    # =====================================================
    fig_reg = plot_dataset_figures(
        y_scores_dict=y_scores_reg,
        y_true_dict=y_true_reg,
        model_colors=model_colors,
        title="2020–2021 Regular Season — Model Comparison",
        results=results
    )

    fig_po = plot_dataset_figures(
        y_scores_dict=y_scores_po,
        y_true_dict=y_true_po,
        model_colors=model_colors,
        title="2020–2021 Playoffs — Model Comparison",
        results=results
    )

    wandb.log({
        "Regular_2020_21_fig": wandb.Image(fig_reg),
        "Playoffs_2020_21_fig": wandb.Image(fig_po),
    })

    wandb.finish()


# %%
if __name__ == "__main__":
    main()