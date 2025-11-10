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
    
    # Limiter top_n au nombre de features disponibles
    n_features = len(feature_names)
    top_n = min(top_n, n_features)
    
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


def plot_learning_curves(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    title="Learning Curves",
    save_path=None
):
    """
    Plot les courbes d'apprentissage (train vs validation loss).
    
    Args:
        model: Modèle avec attribut evals_result_ (XGBoost)
        X_train, y_train: Données d'entraînement
        X_val, y_val: Données de validation
        title: Titre du graphique
        save_path: Chemin pour sauvegarder
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Si le modèle a des résultats d'évaluation (XGBoost)
    if hasattr(model, 'evals_result'):
        results = model.evals_result()
        if results and len(results) > 0:
            # Obtenir la première clé pour les epochs
            first_key = list(results.keys())[0]
            epochs = len(results[first_key]['logloss'])
            x_axis = range(0, epochs)
            
            # Loss curves - tracer toutes les courbes disponibles
            for key in results.keys():
                label = 'Train' if 'validation_0' in key else 'Validation'
                axes[0].plot(x_axis, results[key]['logloss'], label=label)
            
            axes[0].set_xlabel('Iteration')
            axes[0].set_ylabel('Log Loss')
            axes[0].set_title('Loss Curves')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        else:
            # Pas de résultats d'évaluation, afficher un message
            axes[0].text(0.5, 0.5, 'No training history available', 
                        ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title('Loss Curves')
    
    # Entraîner avec différentes tailles de données
    from sklearn.model_selection import learning_curve
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        random_state=42
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    axes[1].plot(train_sizes, train_mean, label='Train', marker='o')
    axes[1].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    axes[1].plot(train_sizes, val_mean, label='Validation', marker='o')
    axes[1].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
    axes[1].set_xlabel('Training Size')
    axes[1].set_ylabel('AUC Score')
    axes[1].set_title('Learning Curve (AUC vs Training Size)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix_detailed(
    y_true,
    y_pred,
    y_pred_proba=None,
    threshold=0.5,
    title="Confusion Matrix",
    save_path=None
):
    """
    Affiche une matrice de confusion détaillée avec métriques.
    
    Args:
        y_true: Vraies labels
        y_pred: Prédictions binaires
        y_pred_proba: Probabilités prédites (optionnel)
        threshold: Seuil de décision
        title: Titre
        save_path: Chemin pour sauvegarder
    """
    from sklearn.metrics import confusion_matrix, classification_report
    
    # Si probabilités fournies, recalculer les prédictions avec le seuil
    if y_pred_proba is not None:
        y_pred = (y_pred_proba >= threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Matrice de confusion (counts)
    im1 = axes[0].imshow(cm, cmap='Blues', alpha=0.7)
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(['No Goal (0)', 'Goal (1)'])
    axes[0].set_yticklabels(['No Goal (0)', 'Goal (1)'])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title(f'Confusion Matrix (Counts)\nThreshold = {threshold}')
    
    # Annoter avec counts et pourcentages
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = count / cm.sum() * 100
            axes[0].text(j, i, f'{count:,}\n({pct:.1f}%)',
                        ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black',
                        fontweight='bold', fontsize=12)
    
    # Matrice de confusion (normalized)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im2 = axes[1].imshow(cm_norm, cmap='Reds', alpha=0.7, vmin=0, vmax=1)
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(['No Goal (0)', 'Goal (1)'])
    axes[1].set_yticklabels(['No Goal (0)', 'Goal (1)'])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title('Confusion Matrix (Normalized)')
    
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, f'{cm_norm[i, j]:.2%}',
                        ha='center', va='center',
                        color='white' if cm_norm[i, j] > 0.5 else 'black',
                        fontweight='bold', fontsize=12)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Afficher le rapport de classification
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Goal', 'Goal']))
    
    return fig


def plot_threshold_analysis(
    y_true,
    y_pred_proba,
    thresholds=None,
    save_path=None
):
    """
    Analyse l'impact du seuil de décision sur les métriques.
    
    Args:
        y_true: Vraies labels
        y_pred_proba: Probabilités prédites
        thresholds: Liste des seuils à tester (si None, 100 points entre 0 et 1)
        save_path: Chemin pour sauvegarder
    """
    from sklearn.metrics import precision_recall_curve, f1_score
    
    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)
    
    precisions = []
    recalls = []
    f1_scores = []
    accuracies = []
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        
        # Calculer les métriques
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        accuracies.append(accuracy)
    
    # Trouver le seuil optimal (maximise F1)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Precision, Recall, F1 vs Threshold
    axes[0, 0].plot(thresholds, precisions, label='Precision', linewidth=2)
    axes[0, 0].plot(thresholds, recalls, label='Recall', linewidth=2)
    axes[0, 0].plot(thresholds, f1_scores, label='F1-Score', linewidth=2)
    axes[0, 0].axvline(optimal_threshold, color='red', linestyle='--', 
                       label=f'Optimal={optimal_threshold:.3f}')
    axes[0, 0].set_xlabel('Threshold')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Metrics vs Threshold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Precision-Recall Curve
    axes[0, 1].plot(recalls, precisions, linewidth=2)
    axes[0, 1].scatter([recalls[optimal_idx]], [precisions[optimal_idx]], 
                       color='red', s=100, zorder=5, label='Optimal')
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. F1-Score vs Threshold (zoom)
    axes[1, 0].plot(thresholds, f1_scores, linewidth=2, color='green')
    axes[1, 0].axvline(optimal_threshold, color='red', linestyle='--', linewidth=2)
    axes[1, 0].axvline(0.5, color='gray', linestyle=':', linewidth=2, label='Default=0.5')
    axes[1, 0].scatter([optimal_threshold], [f1_scores[optimal_idx]], 
                       color='red', s=100, zorder=5)
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].set_title(f'F1-Score vs Threshold (Max={f1_scores[optimal_idx]:.4f})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Accuracy vs Threshold
    axes[1, 1].plot(thresholds, accuracies, linewidth=2, color='purple')
    axes[1, 1].axvline(optimal_threshold, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Threshold')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Accuracy vs Threshold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Threshold Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    print(f"\n Optimal Threshold: {optimal_threshold:.4f}")
    print(f"  Precision: {precisions[optimal_idx]:.4f}")
    print(f"  Recall:    {recalls[optimal_idx]:.4f}")
    print(f"  F1-Score:  {f1_scores[optimal_idx]:.4f}")
    
    return fig, optimal_threshold


def compare_models_detailed(
    models_dict,
    X_val,
    y_val,
    save_path=None
):
    """
    Comparaison détaillée de plusieurs modèles avec graphiques.
    
    Args:
        models_dict: Dict {name: (model, features or None)}
        X_val: Features de validation
        y_val: Labels de validation
        save_path: Chemin pour sauvegarder
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    
    results = []
    
    for name, (model, features) in models_dict.items():
        X_subset = X_val[features] if features else X_val
        y_pred_proba = model.predict_proba(X_subset)[:, 1]
        y_pred = model.predict(X_subset)
        
        metrics = {
            'Model': name,
            'AUC-ROC': roc_auc_score(y_val, y_pred_proba),
            'AP': average_precision_score(y_val, y_pred_proba),
            'Brier': brier_score_loss(y_val, y_pred_proba),
            'Accuracy': accuracy_score(y_val, y_pred),
            'Precision': precision_score(y_val, y_pred, zero_division=0),
            'Recall': recall_score(y_val, y_pred, zero_division=0),
            'F1': f1_score(y_val, y_pred, zero_division=0),
            'N_features': len(features) if features else X_val.shape[1]
        }
        results.append(metrics)
    
    df = pd.DataFrame(results)
    
    # Afficher le tableau
    print("\n" + "="*100)
    print("COMPARAISON DETAILLEE DES MODELES")
    print("="*100)
    print(df.to_string(index=False))
    
    # Identifier les meilleurs
    print(f"\n Meilleur AUC-ROC: {df.loc[df['AUC-ROC'].idxmax(), 'Model']}")
    print(f" Meilleur F1:      {df.loc[df['F1'].idxmax(), 'Model']}")
    print(f" Moins de features: {df.loc[df['N_features'].idxmin(), 'Model']} ({df['N_features'].min()} features)")

    # Graphique de comparaison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics_to_plot = [
        ('AUC-ROC', 'Higher is better'),
        ('F1', 'Higher is better'),
        ('Brier', 'Lower is better'),
        ('N_features', 'Lower is better')
    ]
    
    for idx, (metric, desc) in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        bars = ax.barh(df['Model'], df[metric], color='steelblue', alpha=0.7)
        ax.set_xlabel(metric, fontsize=12)
        ax.set_title(f'{metric} - {desc}', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Annoter les valeurs
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f' {df[metric].iloc[i]:.4f}' if metric != 'N_features' else f' {int(df[metric].iloc[i])}',
                   ha='left', va='center', fontweight='bold')
    
    plt.suptitle('Model Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return df, fig
