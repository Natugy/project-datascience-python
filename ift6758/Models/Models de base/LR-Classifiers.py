import sys
import os
from math import sqrt
from matplotlib.ticker import PercentFormatter
import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score,roc_auc_score,roc_curve
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve, CalibrationDisplay
import matplotlib.ticker as ticker
import wandb

from matplotlib.backends.backend_pdf import PdfPages

# chemin absolu de la racine du projet
current_file = Path(__file__).absolute()
project_root = current_file.parent.parent.parent.parent  # Remonte 4 niveaux
sys.path.insert(0, str(project_root))

from ift6758.data.pandas_conversion import get_dataframe_from_data

# cle de wandb
api_key= os.environ.get("WANDB_API_KEY") 
wandb.login(key="c0237e1d7cc9e5b7d3a6ecf6d37074b81d5298be")

"""if api_key:
    try:
        wandb.login(key=api_key)  
        print('Wandb authentifié avec succès')
    except Exception as e:
        print(f"Erreur lors du login wandb: {e}")"""
       

"""
créer un dataframe avec les caracteristiques
-Distance du filet
- Angle relatif au filet
- est un but (0 ou 1)
- Filet vide (0 ou 1, vous pouvez supposons que les NaN sont 0)
"""
def create_feature_dataframe(df,feature='all'):
    # Calculer la distance au filet pour chaque tir
    def calculate_distance(shot_row):
        goal_x = 89
        goal_y = 0
        normalized_x = abs(shot_row['xCoord'])
        normalized_y = abs(shot_row['yCoord'])
        distance = sqrt((normalized_x - goal_x)**2 + (normalized_y - goal_y)**2)
        return distance

    # Calculer l'angle relatif au filet pour chaque tir
    def calculate_angle(shot_row):
        normalized_x = abs(shot_row['xCoord'])
        normalized_y = abs(shot_row['yCoord'])
        angle = np.arctan2(normalized_y, (89 - normalized_x)) * (180 / np.pi)
        return angle

    if feature == 'distance':
        df['distance_to_goal'] = df.apply(calculate_distance, axis=1)
        df['is_goal'] = df['typeDescKey'].apply(lambda x: 1 if x == 'goal' else 0)
    elif feature == 'angle':
        df['angle_to_goal'] = df.apply(calculate_angle, axis=1)
        df['is_goal'] = df['typeDescKey'].apply(lambda x: 1 if x == 'goal' else 0)
    elif feature == 'all':
        # Appliquer les calculs et créer le DataFrame des caractéristiques
        df['distance_to_goal'] = df.apply(calculate_distance, axis=1)
        df['angle_to_goal'] = df.apply(calculate_angle, axis=1)
        df['is_goal'] = df['typeDescKey'].apply(lambda x: 1 if x == 'goal' else 0)
        #df['is_empty_net'] = df['emptyNet'].fillna(0)

    return df


""" 
creation des datasets d'entrainement, de validation et de test
donneees de test : 2020-2021
donneees de validation : 2019-2020
donneees d'entrainement : toutes les autres saisons avant 2016-2017 jusqu'a 2018-2019
"""

def create_datasets(df,feature='all'):
    # Filtrer les données pour chaque ensemble   
    df_copy = df.copy()
    df_copy['season'] = df_copy['season'].astype(str)
    train_df = df_copy[df_copy['season'].isin(['20162017', '20172018', '20182019'])]
    val_df = df_copy[df_copy['season'] == '20192020']
    test_df = df_copy[df_copy['season'] == '20202021']

    # Séparer les caractéristiques et les étiquettes
    if feature == 'distance':
        X_train = train_df[['distance_to_goal']]       
        X_val = val_df[['distance_to_goal']]      
        X_test = test_df[['distance_to_goal']]
       
    elif feature== 'angle':
        X_train = train_df[['angle_to_goal']]       
        X_val = val_df[['angle_to_goal']]        
        X_test = test_df[['angle_to_goal']]
        
    elif feature == 'all':
        # avec toutes les caracteristiques : distance, angle et empty net
        X_train = train_df[['distance_to_goal', 'angle_to_goal']]
        X_val = val_df[['distance_to_goal', 'angle_to_goal']]      
        X_test = test_df[['distance_to_goal', 'angle_to_goal']]
        
    y_train = train_df['is_goal']
    y_val = val_df['is_goal']
    y_test = test_df['is_goal']

    return X_train, y_train, X_val, y_val, X_test, y_test



# Fonction pour entraîner un classificateur de régression logistique
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

# Fonction pour évaluer le modèle, avec toutes les métriques
def evaluate_accuracy_model(model, X, y):
    accuracy = model.score(X, y)
    return accuracy

# calcul les probabilités de prédictions
def get_predicted_probabilities(model, X):
    y_pred_proba = model.predict_proba(X)[:, 1]
    return y_pred_proba

def find_best_threshold(y_true, y_scores):
    
    #Trouve le seuil optimal qui maximise le F1-score    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Calculer le F1-score pour chaque seuil
    f1_scores = []
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        if len(np.unique(y_pred)) > 1:  # vérifie qu'il y a au moins 2 classes différentes 
            f1_scores.append(f1_score(y_true, y_pred))
        else:
            f1_scores.append(0)    
    # Trouver le seuil qui maximise le F1-score
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    return best_threshold

def evaluate_metrics_model(model, X, y):    
    # Obtenir les probabilités prédites
    y_pred_proba = get_predicted_probabilities(model, X)
    
    # Si aucun seuil n'est fourni, trouver le meilleur seuil
    
    threshold = find_best_threshold(y, y_pred_proba)
    
    # Appliquer le seuil pour les prédictions
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Vérifier qu'il y a au moins 2 classes dans les prédictions
    if len(np.unique(y_pred)) > 1:
        # Calculer les métriques
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
    else:
        # Si toutes les prédictions sont identiques
        precision = recall = f1 = 0.0
    
    return precision, recall, f1
     
     


def afficher_graphiques_metrics(y_true,y_scores, model):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    axes = axes.flatten() 
    # mettre un titre a figure 
    fig.suptitle(f"Métriques du modèle {model}", fontsize=16, fontweight='bold')
    # graphique ROC
    axes[0].plot(fpr, tpr, label='ROC Curve')
    axes[0].plot([0, 1], [0, 1], 'k--', label='Random Classifier')    
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('Receiver Operating Characteristic (ROC) Curve')
    axes[0].legend()
    
    #graphique taux de buts comme une fonction du centile de la probabilité de tir donnée par le modèle
    axes[1].set_title("Taux de buts centile")
    # Trier les scores et labels ensemble avec les mêmes indices
    sorted_indices = np.argsort(y_scores)[::-1]  # Indices triés décroissant
    sorted_scores = y_scores[sorted_indices]
    sorted_labels = y_true.values[sorted_indices] if hasattr(y_true, 'values') else y_true[sorted_indices]

    # Créer des bins de centiles pour réduire le nombre de points
    n_bins = 10  
    n = len(sorted_scores)
    bin_size = n // n_bins
    
    centiles_list = []
    taux_buts_list = []

    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < n_bins - 1 else n
        bin_labels = sorted_labels[start_idx:end_idx]        
        # Taux de buts dans ce bin = moyenne des labels (1 = but, 0 = non-but)
        taux_but = np.mean(bin_labels)        
        # Centile moyen du bin 
        centile_moyen = 100 - (i + 0.5) * (100 / n_bins)      
        centiles_list.append(centile_moyen)
        taux_buts_list.append(taux_but)
        axes[1].plot(centiles_list, taux_buts_list, marker='o', linewidth=1.5)
        axes[1].yaxis.set_major_formatter(ticker.PercentFormatter(1))
        axes[1].set_xlim(100, 0)
        axes[1].set_ylim(0, 1)
        axes[1].set_xlabel('Centile de la probabilité de tir')
        axes[1].set_ylabel('Taux de buts')
        axes[1].legend()

    #graphique proportion cumulée de buts comme une fonction du centile de la probabilité de tir donnée par le modèle
    axes[2].set_title("Proportion cumulée centile")       
    # Calculer le nombre total de buts
    total_buts = np.sum(sorted_labels == 1)       
    # Nombre cumulatif de buts jusqu'à chaque position
    if total_buts != 0:
        cumsum_buts = np.cumsum(sorted_labels == 1)   
        # Proportion cumulée = cumsum_buts / total_buts
        proportion_cumulee = cumsum_buts / total_buts
        # Calcul des centiles
        n = len(sorted_scores)
        centiles = np.arange(1, n+1) / n * 100
        axes[2].plot(centiles, proportion_cumulee, marker='o', color='orange')
        axes[2].yaxis.set_major_formatter(ticker.PercentFormatter(1))       
        axes[2].set_xlabel('Centile de la probabilité de tir')
        axes[2].set_ylabel('Proportion cumulée de buts')
        axes[2].legend()

    #graphique diagramme de fiabilité (courbe de calibration)
    axes[3].set_title("Diagramme de fiabilité (courbe de calibration)")
    CalibrationDisplay.from_predictions(y_true, y_scores, n_bins=10, ax=axes[3])
    axes[3].plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    axes[3].set_xlabel('Predicted Probability') 
    axes[3].set_ylabel('True Probability')
    axes[3].legend()
    plt.tight_layout()

    #Logger la figure dans wandb
    wandb.log({f"graphs/{model.replace(' ', '_')}": wandb.Image(fig)})
    
    return fig

def generate_random_baseline(y_true):    
    np.random.seed(42)  # Pour reproductibilité
    y_scores_random = np.random.uniform(0, 1, size=len(y_true))
    return y_scores_random

# main function pour tester le code
def main():

    wandb.init(
        project="IFT6758-2025",
        name="LR-Classifier-Models"
    )
    # Charger les données (assurez-vous que df est défini)
    saisons = ['20162017', '20172018', '20182019', '20192020', '20202021']
    df = pd.DataFrame()
    for saison in saisons:
            df = pd.concat([df, get_dataframe_from_data(saison)])   

    figures =[]
    # 1 : Régression logistique, entraînée sur la distance uniquement

    # Créer le DataFrame des caractéristiques
    feature_df_distance = create_feature_dataframe(df,'distance')    
    
    #créer les datasets pou chaque model
    X_train_distance, y_train, X_val_distance, y_val, X_test_distance, y_test = create_datasets(feature_df_distance,'distance')    
    # Entraîner modèle avec distance uniquement
    model_distance = train_logistic_regression(X_train_distance, y_train)
    
    # Évaluer chaque modèle sur les données de validation
    val_accuracy_distance = evaluate_accuracy_model(model_distance, X_val_distance, y_val)
    print(f'Validation Accuracy (Distance): {val_accuracy_distance:.4f}')
    
    # Regardez les prédictions et interprétez-les pour chaque modèle
    y_pred_distance = model_distance.predict(X_val_distance)
    print(f'Predictions on validation set (Distance): {y_val.values[:100]} {y_pred_distance[:100]}')  
        
    # Calculer et afficher les métriques
    precision, recall, f1 = evaluate_metrics_model(model_distance, X_val_distance, y_val)
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')
    auc_score = roc_auc_score(y_val, get_predicted_probabilities(model_distance, X_val_distance))
    # logger les metriques dans wandb
    wandb.log({
        "model_distance/accuracy": val_accuracy_distance,
        "model_distance/precision": precision,
        "model_distance/recall": recall,
        "model_distance/f1": f1,
        "model_distance/auc": auc_score,
    })

    figures.append(afficher_graphiques_metrics(y_val, get_predicted_probabilities(model_distance, X_val_distance),'LR distance'))

    #2 Régression logistique, entraînée sur l' angle uniquement
    
    # Créer le DataFrame des caractéristiques
    feature_df_angle = create_feature_dataframe(df,'angle')
  
    #créer les datasets pou chaque model
    X_train_angle, _, X_val_angle,_, X_test_angle, _ = create_datasets(feature_df_angle,'angle')
      
    # Entraîner modèle avec angle uniquement
    model_angle = train_logistic_regression(X_train_angle, y_train)
   
    # Évaluer chaque modèle sur les données de validation    
    val_accuracy_angle = evaluate_accuracy_model(model_angle, X_val_angle, y_val)
    print(f'Validation Accuracy (Angle): {val_accuracy_angle:.4f}')

    # Regardez les prédictions et interprétez-les pour chaque modèle   
    y_pred_angle = model_angle.predict(X_val_angle)
    print(f'Predictions on validation set (Angle): {y_val.values[:100]} {y_pred_angle[:100]}')

    # Calculer et afficher les métriques
    precision, recall, f1 = evaluate_metrics_model(model_angle, X_val_angle, y_val)
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')
    auc_score = roc_auc_score(y_val, get_predicted_probabilities(model_angle, X_val_angle))
    wandb.log({
        "model_angle/accuracy": val_accuracy_angle,
        "model_angle/precision": precision,
        "model_angle/recall": recall,
        "model_angle/f1": f1,
        "model_angle/auc": auc_score,
    })
     
    figures.append(afficher_graphiques_metrics(y_val, get_predicted_probabilities(model_angle, X_val_angle),'LR angle'))

    # 3 : Régression logistique, entraînée sur la distance et l' angle
    
    # Créer le DataFrame des caractéristiques
    feature_df = create_feature_dataframe(df,'all')

    #créer les datasets pou chaque model
    X_train_all, _, X_val_all, y_val_all, X_test_all, y_test_all = create_datasets(feature_df,'all')

    # Entraîner modèle avec distance et angle
    model_all = train_logistic_regression(X_train_all, y_train)

    # Évaluer chaque modèle sur les données de validation
    val_accuracy_all = evaluate_accuracy_model(model_all, X_val_all, y_val)
    print(f'Validation Accuracy (All Features): {val_accuracy_all:.4f}')

    # Regardez les prédictions et interprétez-les pour chaque modèle
    y_pred_all = model_all.predict(X_val_all)
    print(f'Predictions on validation set (All Features): {y_val.values[:100]} {y_pred_all[:100]}')

    # Calculer et afficher les métriques
    precision, recall, f1 = evaluate_metrics_model(model_all, X_val_all, y_val)
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')
    auc_score = roc_auc_score(y_val, get_predicted_probabilities(model_all, X_val_all))
    wandb.log({
        "model_distanceangle/accuracy": val_accuracy_all,
        "model_distanceangle/precision": precision,
        "model_distanceangle/recall": recall,
        "model_distanceangle/f1": f1,
        "model_distanceangle/auc": auc_score,
    })
   
    figures.append(afficher_graphiques_metrics(y_val, get_predicted_probabilities(model_all, X_val_all), 'LR distance et angle'))

    # Générer la ligne de base aléatoire U(0,1)
    #y_scores_random = generate_random_baseline(y_val)

    # Sauvegarder les 3 figures dans un pdf
    output_dir = Path(__file__).resolve().parent.parent.parent.parent / "figures" / "milestone2"
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "LR_Classifiers.pdf"
    print(f"\n Sauvegarde du PDF dans: {pdf_path}")
    
    with PdfPages(pdf_path) as pdf:
        for i, fig in enumerate(figures, 1):
            pdf.savefig(fig, bbox_inches='tight')
            print(f"Page {i} ajoutée au PDF")
    
    print(f"PDF sauvegardé avec succès: {pdf_path}")
    print(f"Nombre de pages: {len(figures)}")
    
    # Fermer wandb
    wandb.finish()
   
if __name__ == "__main__":
    main()    