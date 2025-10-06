import sys
import os
from math import sqrt
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns 



def afficher_graphique_typestir(df, saison):
    # grouper par type de tir, agregé sur tous les tirs
    shots_types = df.groupby('shotType').size().reset_index(name='shots')
    goals_types = df[df['typeDescKey'] == 'goal'].groupby('shotType').size().reset_index(name='goals')

    # Fusionner les données AVANT le tri
    merged_data = pd.merge(shots_types, goals_types, on='shotType', how='left')
    merged_data['goals'] = merged_data['goals'].fillna(0)

    # Créer le graphique en barres superposées
    fig, ax = plt.subplots(figsize=(12, 8))

    # Trier la version fusionnée
    merged_data = merged_data.sort_values('shots', ascending=False) 
    #print('Data merget',merged_data)

    x_pos = np.arange(len(shots_types))
    shot_types = merged_data['shotType']

    # Barre de base : tous les tirs (fond clair)
    bars_shots = ax.bar(x_pos, merged_data['shots'], 
                   color='y', alpha=0.6, 
                   label='Tirs', edgecolor='black', linewidth=0.5)

    # Barre superposée 
    bars_goals = ax.bar(x_pos, merged_data['goals'], 
                   color='g', alpha=1.0, 
                   label='Buts', edgecolor='black', linewidth=0.5)

    # Configuration du graphique
    ax.set_title(f'Comparaison des Types de Tirs\n(Buts/tirs superposés - Saison {saison}', 
             fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Types de Tirs', fontsize=12)
    ax.set_ylabel('Nombre Tirs', fontsize=12)
    # Étiquettes sur l'axe X
    ax.set_xticks(x_pos)
    ax.set_xticklabels(shot_types, rotation=45, ha='right')
    # Légende
    ax.legend(loc='upper right')
    # Grille pour faciliter la lecture
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    #afficher les etiquettes pour voir les types de tir les plus courant
    #ax.bar_label(bars_shots, fmt='%.0f', padding=3)
    #ax.bar_label(bars_goals, fmt='%.0f', padding=3)

   # afficher les pourcentages pour voir les type de tir les plus dangereux
    merged_data['success_rate'] = (merged_data['goals'] / merged_data['shots'] * 100).round(1)
    ax.bar_label(bars_goals, labels=[f'{r}%' for r in merged_data['success_rate']], padding=3)
    plt.tight_layout()
    plt.show()     
     


# Calculer la distance de tir pour chaque événement
# la distance euclidienne, la distance ecuclidienne est calculé par rapport aux coordonnées des buts 
# Les coordonnées du filet sont généralement à (89, 0) ou (-89, 0) selon le côté
def calculate_distance(shot_row):
    # Filet à x=89 pour les tirs vers le filet adverse    
    goal_x = 89
    goal_y = 0    
    # Prendre la valeur absolue de x pour normaliser vers le filet positif
    normalized_x = abs(shot_row['xCoord'])
    normalized_y = abs(shot_row['yCoord'])

    distance = int(sqrt((normalized_x - goal_x)**2 + (normalized_y - goal_y)**2))
    return distance

"""  """
def afficher_graphique_distance_but(df,saison):
    
    # Appliquer le calcul de distance. ajouter une colone au dataframe
    df['distance_to_goal'] = df.apply(calculate_distance, axis=1)

    # Créer des intervalles de distance (bins)
    # ca assure une stabilité statistique pour la definition des probabilités
    # la probabilité d'un tir a partir une distance x d'etre un but peut-être 0 ou 1  
    df['distance_bin'] = pd.cut(df['distance_to_goal'], bins=10, precision=0)
    
    # Calculer la probabilité de but pour chaque intervalle de distance
    distance_analysis = df.groupby(['distance_bin', 'typeDescKey']).size().reset_index(name='count')
    distance_pivot = distance_analysis.pivot(index='distance_bin', columns='typeDescKey', values='count').fillna(0)
    distance_pivot['probability'] = distance_pivot.get('goal', 0) / (distance_pivot.get('shot-on-goal', 0) + distance_pivot.get('goal', 0)) * 100
 
    # créer graphique avec barres groupées
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 10))    
    distance_pivot['probability'].dropna().plot(kind='bar', ax=ax1, color='darkblue')    
    ax1.set_title(f'Probabilité de But par Distance - Saison {saison}')
    ax1.set_xlabel('Distance au filet')
    ax1.set_ylabel('Probabilité de But (%)')    
    ax1.bar_label(ax1.containers[0], fmt='%.1f%%', padding=3)
    plt.tight_layout()
    plt.show()


""" fonction qui affiche un graphique qui montre le pourcentage de buts
    en fonction de la distance au filet et le type de tir pour une saison 
"""
def afficher_graphique_pourcentage_dist_type(df, saison):
    
    df['distance_to_goal'] = df.apply(calculate_distance, axis=1)
    df['distance_bin'] = pd.cut(df['distance_to_goal'], bins=10, precision=0)

    # Obtenir tous les types de tirs
    all_shot_types = df['shotType'].value_counts().index
    
    # Créer un graphique pour chaque type de tir
    for shot_type in all_shot_types:
        # Filtrer pour ce type de tir
        shot_data = df[df['shotType'] == shot_type]
        
        # Créer pivot pour ce type de tir
        type_pivot = shot_data.groupby(['distance_bin', 'typeDescKey'], observed=True).size().reset_index(name='count')
        type_pivot = type_pivot.pivot(index='distance_bin', columns='typeDescKey', values='count').fillna(0)    
        type_pivot['percentage'] = (type_pivot.get('goal', 0) / (type_pivot.get('shot-on-goal', 0) + type_pivot.get('goal', 0)) * 100)
        
        # Créer un graphique individuel
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        type_pivot['percentage'].plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title(f'Pourcentage de Buts - {shot_type} (Saison {saison})', fontsize=14, fontweight='bold')
        ax.set_ylabel('Pourcentage de Buts (%)')
        ax.set_xlabel('Distance au Filet')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        ax.bar_label(ax.containers[0], fmt='%.1f%%', padding=3)
        plt.tight_layout()
        plt.show()



def afficher_graphique_pourcentage_dist_types(df, saison):
    df['distance_to_goal'] = df.apply(calculate_distance, axis=1)
    df['distance_bin'] = pd.cut(df['distance_to_goal'], bins=10, precision=0)

    # 2) Agrégation par (distance_bin, shotType, typeDescKey)
    grouped = (
        df.groupby(['distance_bin', 'shotType', 'typeDescKey'], observed=True)
        .size()
        .reset_index(name='count')
        )

    # 3) Pivot pour séparer buts et tirs cadrés par bin et type
    pivot = (
    grouped.pivot_table(index=['distance_bin', 'shotType'],
                                columns='typeDescKey',
                                values='count',
                                fill_value=0)
                             .reset_index()
        )

    # 4) Pourcentage = buts / (tirs cadrés + buts)
    pivot['percentage'] = (
            pivot.get('goal', 0) /
            (pivot.get('shot-on-goal', 0) + pivot.get('goal', 0)).replace(0, np.nan) * 100
        )

    # 5) Table finale distance (lignes) x type (colonnes)
    heat = pivot.pivot(index='distance_bin', columns='shotType', values='percentage')

    try:
             bin_labels = [f"{int(iv.left)}-{int(iv.right)}" for iv in heat.index]  # utiliser l’index présent
             heat.index = bin_labels
    except AttributeError:
        # si l’index n’est pas IntervalIndex (déjà des str), on ne fait rien
         pass

    # 6) Labels de bins entiers
    if isinstance(heat.index.dtype, pd.CategoricalDtype) or str(heat.index.dtype) == 'category':
            bin_labels = [f"{int(iv.left)}-{int(iv.right)}" for iv in heat.index.categories]
            heat.index = bin_labels

    # 7) Heatmap
    plt.figure(figsize=(12, 7))
    ax = sns.heatmap(
            heat,
            annot=True,
            fmt='.1f',
            cmap='YlGnBu',
            linewidths=.5,
            linecolor='white',
            cbar_kws={'label': '% de buts'}
        )
    ax.set_title(f'% de buts par distance et type de tir — Saison {saison}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Type de tir')
    ax.set_ylabel('Distance au filet (pieds)')
    plt.tight_layout()
    plt.show()        