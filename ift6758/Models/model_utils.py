"""
Fonctions utilitaires pour l'entraînement et l'évaluation des modèles.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                             recall_score, f1_score, confusion_matrix)
from typing import Dict, Any


def evaluate_model(model, X, y, features=None, model_name="Model"):
    """
    Évalue un modèle et retourne les métriques.
    
    Args:
        model: Modèle entraîné
        X: Features
        y: Labels
        features: Liste des features à utiliser (si None, utilise toutes)
        model_name: Nom du modèle pour l'affichage
    
    Returns:
        Dict avec les métriques
    """
    X_subset = X[features] if features is not None else X
    y_pred_proba = model.predict_proba(X_subset)[:, 1]
    y_pred = model.predict(X_subset)
    
    metrics = {
        'AUC': roc_auc_score(y, y_pred_proba),
        'Accuracy': accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred, zero_division=0),
        'Recall': recall_score(y, y_pred, zero_division=0),
        'F1': f1_score(y, y_pred, zero_division=0),
        'model_name': model_name
    }
    
    return metrics


def print_metrics(metrics, show_confusion=False, cm=None):
    """Affiche les métriques de manière formatée."""
    import sys
    sys.stdout.flush()
    
    print(f"\n{'='*60}")
    print(f"METRIQUES: {metrics['model_name']}")
    print(f"{'='*60}")
    print(f"  AUC-ROC:    {metrics['AUC']:.4f}")
    print(f"  Accuracy:   {metrics['Accuracy']:.4f}")
    print(f"  Precision:  {metrics['Precision']:.4f}")
    print(f"  Recall:     {metrics['Recall']:.4f}")
    print(f"  F1-Score:   {metrics['F1']:.4f}")
    
    if show_confusion and cm is not None:
        print(f"\nMatrice de confusion:")
        print(f"  TN: {cm[0,0]:,} | FP: {cm[0,1]:,}")
        print(f"  FN: {cm[1,0]:,} | TP: {cm[1,1]:,}")
    
    sys.stdout.flush()


def log_metrics_to_wandb(model, X_train, y_train, X_val, y_val, prefix="model", wandb_run=None):
    """
    Log les métriques d'évaluation sur Wandb.
    
    Args:
        model: Modèle entraîné
        X_train: Features d'entraînement
        y_train: Labels d'entraînement
        X_val: Features de validation
        y_val: Labels de validation
        prefix: Préfixe pour les métriques
        wandb_run: Instance de wandb.run (si None, utilise wandb direct)
    """
    from sklearn.metrics import roc_auc_score, brier_score_loss
    
    # Prédictions
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Métriques
    metrics = {
        f"{prefix}/train_auc": roc_auc_score(y_train, y_train_pred_proba),
        f"{prefix}/val_auc": roc_auc_score(y_val, y_val_pred_proba),
        f"{prefix}/train_brier": brier_score_loss(y_train, y_train_pred_proba),
        f"{prefix}/val_brier": brier_score_loss(y_val, y_val_pred_proba),
    }
    
    if wandb_run:
        import wandb
        wandb.log(metrics)
    
    print(f"\nMetrics ({prefix}):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    return metrics

def plot_feature_importance(model, feature_names, title="Feature Importance", top_n=10):
    """Plot les top N features importantes."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), importances[indices], color='steelblue', alpha=0.7)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    return fig


def compare_models(metrics_list):
    """
    Compare plusieurs modèles et affiche un tableau.
    
    Args:
        metrics_list: Liste de dictionnaires de métriques
    
    Returns:
        DataFrame avec la comparaison
    """
    df = pd.DataFrame(metrics_list)
    cols = ['model_name', 'AUC', 'Accuracy', 'Precision', 'Recall', 'F1']
    df = df[cols]
    
    print(f"\n{'='*80}")
    print("COMPARAISON DES MODELES")
    print(f"{'='*80}\n")
    print(df.to_string(index=False))
    
    # Identifier les meilleurs
    best_auc = df.loc[df['AUC'].idxmax(), 'model_name']
    best_f1 = df.loc[df['F1'].idxmax(), 'model_name']
    
    print(f"\nMeilleur AUC: {best_auc}")
    print(f"Meilleur F1:  {best_f1}")
    
    return df


def plot_models_comparison(metrics_list, save_path=None):
    """Visualise la comparaison des modèles."""
    df = pd.DataFrame(metrics_list)
    metrics_to_plot = ['AUC', 'Precision', 'Recall', 'F1']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        bars = ax.bar(range(len(df)), df[metric], color=colors[idx], alpha=0.7)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels([name[:20] for name in df['model_name']], rotation=15, ha='right')
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric} - Comparaison', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Ajouter valeurs
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{df[metric].iloc[i]:.4f}',
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure sauvegardee: {save_path}")
    
    plt.show()
    return fig


def apply_scaler(X, scaler, fit=False):
    """
    Applique un scaler aux donnees.
    
    Args:
        X: DataFrame ou array de features
        scaler: Instance de StandardScaler ou autre scaler sklearn
        fit: Si True, fit_transform. Si False, transform seulement
        
    Returns:
        DataFrame ou array scale
    """
    if scaler is None:
        return X
    
    is_dataframe = isinstance(X, pd.DataFrame)
    
    if fit:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    # Conserver le format DataFrame si applicable
    if is_dataframe:
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    return X_scaled
