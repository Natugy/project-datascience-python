"""
Partie 2 - Génération des visualisations pour l'ingénierie des caractéristiques.
Questions 1, 2, et 3.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


def load_training_data() -> pd.DataFrame:
    """Charge les données d'entraînement depuis data/processed/."""
    project_root = Path(__file__).parent.parent.parent
    train_path = project_root / "data" / "processed" / "train_data.csv"
    
    if not train_path.exists():
        raise FileNotFoundError(f"Fichier train_data.csv introuvable: {train_path}")
    
    df = pd.read_csv(train_path)
    print(f"Données chargées: {len(df):,} tirs, {df['is_goal'].sum():,} buts")
    return df


def question1_histograms(df: pd.DataFrame, output_dir: Optional[Path] = None):
    """
    Question 1: Histogrammes de distance et angle, séparés par buts/non-buts + histogramme 2D.
    
    Args:
        df: DataFrame avec les features
        output_dir: Dossier de sortie pour les figures
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "figures" / "milestone2"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("Histogrammes distance et angle")
    print("="*70)
    
    # Figure 1: Histogramme de distance
    fig, ax = plt.subplots(figsize=(12, 7))
    
    goals = df[df['is_goal'] == 1]
    no_goals = df[df['is_goal'] == 0]
    
    bins_distance = np.arange(0, 200, 5)
    
    ax.hist(no_goals['distance_net'], bins=bins_distance, alpha=0.6, 
            label=f'Non-buts (n={len(no_goals):,})', color='skyblue', edgecolor='black')
    ax.hist(goals['distance_net'], bins=bins_distance, alpha=0.7, 
            label=f'Buts (n={len(goals):,})', color='orange', edgecolor='black')
    
    ax.set_xlabel('Distance au filet (pieds)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Nombre de tirs', fontsize=13, fontweight='bold')
    ax.set_title('Histogramme: Nombre de tirs par distance (buts vs non-buts)', 
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    fig_path = output_dir / "Q1_histogram_distance.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Sauvegardé: {fig_path}")
    plt.close()
    
    # Figure 2: Histogramme d'angle
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bins_angle = np.arange(0, 90, 3)
    
    ax.hist(no_goals['angle_net'], bins=bins_angle, alpha=0.6, 
            label=f'Non-buts (n={len(no_goals):,})', color='skyblue', edgecolor='black')
    ax.hist(goals['angle_net'], bins=bins_angle, alpha=0.7, 
            label=f'Buts (n={len(goals):,})', color='orange', edgecolor='black')
    
    ax.set_xlabel('Angle relatif au filet (degrés)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Nombre de tirs', fontsize=13, fontweight='bold')
    ax.set_title('Histogramme: Nombre de tirs par angle (buts vs non-buts)', 
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    fig_path = output_dir / "Q1_histogram_angle.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Sauvegardé: {fig_path}")
    plt.close()
    
    # Figure 3: Histogramme 2D (distance vs angle)
    fig, ax = plt.subplots(figsize=(12, 9))
    
    h = ax.hist2d(df['distance_net'], df['angle_net'], 
                  bins=[bins_distance, bins_angle], 
                  cmap='YlOrRd', cmin=1)
    
    cbar = plt.colorbar(h[3], ax=ax)
    cbar.set_label('Nombre de tirs', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Distance au filet (pieds)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Angle relatif au filet (degrés)', fontsize=13, fontweight='bold')
    ax.set_title('Histogramme 2D: Distribution des tirs (distance vs angle)', 
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    fig_path = output_dir / "Q1_histogram_2D.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Sauvegardé: {fig_path}")
    plt.close()


def question2_goal_rate_curves(df: pd.DataFrame, output_dir: Optional[Path] = None):
    """
    Question 2: Taux de but en fonction de la distance et de l'angle.
    
    Args:
        df: DataFrame avec les features
        output_dir: Dossier de sortie
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "figures" / "milestone2"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("Taux de but vs distance/angle")
    print("="*70)
    
    # Figure 1: Taux de but vs distance
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bins_distance = np.arange(0, 200, 5)
    df['distance_bin'] = pd.cut(df['distance_net'], bins=bins_distance)
    
    goal_rate_dist = df.groupby('distance_bin', observed=True)['is_goal'].agg(['mean', 'count'])
    goal_rate_dist = goal_rate_dist[goal_rate_dist['count'] >= 10]  # Filtrer bins avec peu de données
    
    distance_mids = [interval.mid for interval in goal_rate_dist.index]
    
    ax.plot(distance_mids, goal_rate_dist['mean'] * 100, 
            marker='o', linewidth=2.5, markersize=6, color='#e74c3c')
    
    ax.set_xlabel('Distance au filet (pieds)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Taux de but (%)', fontsize=13, fontweight='bold')
    ax.set_title('Taux de but en fonction de la distance au filet', 
                 fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    
    fig_path = output_dir / "Q2_goal_rate_vs_distance.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"✓ Sauvegardé: {fig_path}")
    plt.close()
    
    # Figure 2: Taux de but vs angle
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bins_angle = np.arange(0, 90, 3)
    df['angle_bin'] = pd.cut(df['angle_net'], bins=bins_angle)
    
    goal_rate_angle = df.groupby('angle_bin', observed=True)['is_goal'].agg(['mean', 'count'])
    goal_rate_angle = goal_rate_angle[goal_rate_angle['count'] >= 10]
    
    angle_mids = [interval.mid for interval in goal_rate_angle.index]
    
    ax.plot(angle_mids, goal_rate_angle['mean'] * 100, 
            marker='o', linewidth=2.5, markersize=6, color='#3498db')
    
    ax.set_xlabel('Angle relatif au filet (degrés)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Taux de but (%)', fontsize=13, fontweight='bold')
    ax.set_title('Taux de but en fonction de l\'angle du tir', 
                 fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    
    fig_path = output_dir / "Q2_goal_rate_vs_angle.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Sauvegardé: {fig_path}")
    plt.close()


def question3_empty_net_analysis(df: pd.DataFrame, output_dir: Optional[Path] = None):
    """
    Question 3: Histogramme des buts uniquement, séparés par filet vide/non-vide.
    Vérification de l'intégrité des données.
    
    Args:
        df: DataFrame avec les features
        output_dir: Dossier de sortie
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "figures" / "milestone2"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("Analyse des filets vides et intégrité des données")
    print("="*70)
    
    # Filtrer uniquement les buts
    goals_only = df[df['is_goal'] == 1].copy()
    
    # Séparer par filet vide / non-vide
    empty_net_goals = goals_only[goals_only['empty_net'] == 1]
    non_empty_goals = goals_only[goals_only['empty_net'] == 0]
    
    print(f"\nButs au total: {len(goals_only):,}")
    print(f"Filet vide:     {len(empty_net_goals):,} ({len(empty_net_goals)/len(goals_only)*100:.1f}%)")
    print(f"Filet non-vide: {len(non_empty_goals):,} ({len(non_empty_goals)/len(goals_only)*100:.1f}%)")
    
    # Histogramme
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bins_distance = np.arange(0, 200, 5)
    
    ax.hist(non_empty_goals['distance_net'], bins=bins_distance, alpha=0.7, 
            label=f'Filet non-vide (n={len(non_empty_goals):,})', 
            color='steelblue', edgecolor='black')
    ax.hist(empty_net_goals['distance_net'], bins=bins_distance, alpha=0.7, 
            label=f'Filet vide (n={len(empty_net_goals):,})', 
            color='coral', edgecolor='black')
    
    ax.set_xlabel('Distance au filet (pieds)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Nombre de buts', fontsize=13, fontweight='bold')
    ax.set_title('Histogramme: Buts par distance (filet vide vs non-vide)', 
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.axvline(x=89, color='red', linestyle='--', linewidth=2, alpha=0.5, 
               label='Zone défensive (x > 89 pieds)')
    plt.tight_layout()
    
    fig_path = output_dir / "Q3_empty_net_goals_histogram.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nSauvegardé: {fig_path}")
    plt.close()
    
    # Buts suspects: filet non-vide + distance > 89 pieds (zone défensive)
    suspicious_goals = non_empty_goals[non_empty_goals['distance_net'] > 100]
    
    print(f"\n  Buts suspects (filet non-vide, distance > 100 pieds): {len(suspicious_goals)}")
    
    if len(suspicious_goals) > 0:
        print("\n Événements suspects détectés:")
        for idx, row in suspicious_goals.head(10).iterrows():
            print(f"      Game {row['idGame']}, Distance: {row['distance_net']:.1f} ft, "
                  f"Coords: ({row['xCoord']:.0f}, {row['yCoord']:.0f}), "
                  f"Team: {row['teamAbbr']}")
        
     
    else:
        print(" Aucun but suspect détecté (filet non-vide > 100 pieds)")
    
    # Statistiques supplémentaires
    print("\nStatistiques des buts par distance:")
    print(f"   Distance moyenne (filet non-vide): {non_empty_goals['distance_net'].mean():.1f} ft")
    print(f"   Distance moyenne (filet vide):     {empty_net_goals['distance_net'].mean():.1f} ft")
    print(f"   Distance médiane (filet non-vide): {non_empty_goals['distance_net'].median():.1f} ft")
    print(f"   Distance médiane (filet vide):     {empty_net_goals['distance_net'].median():.1f} ft")
    
    # Distribution par périodes
    print("\n Distribution des buts par période:")
    for period in sorted(goals_only['game_period'].unique()):
        period_goals = goals_only[goals_only['game_period'] == period]
        empty = (period_goals['empty_net'] == 1).sum()
        non_empty = (period_goals['empty_net'] == 0).sum()
        print(f"   Période {period}: {len(period_goals):,} buts "
              f"(vide: {empty}, non-vide: {non_empty})")


def generate_all_visualizations():
    """Génère toutes les visualisations pour Partie 2."""
    print("\n" + "="*80)
    print(" "*25 + "PARTIE 2 - VISUALISATIONS")
    print("="*80)
    
    # Charger les données
    df = load_training_data()
    
    # Question 1: Histogrammes
    question1_histograms(df)
    
    # Question 2: Taux de but
    question2_goal_rate_curves(df)
    
    # Question 3: Filets vides
    question3_empty_net_analysis(df)
    
    print("\n" + "="*80)
    print("✓ TOUTES LES VISUALISATIONS GÉNÉRÉES AVEC SUCCÈS")
    print("="*80)
    print(f"\nFigures sauvegardées dans: figures/milestone2/")
    print("\nFichiers générés:")
    print("  • Q1_histogram_distance.png")
    print("  • Q1_histogram_angle.png")
    print("  • Q1_histogram_2D.png")
    print("  • Q2_goal_rate_vs_distance.png")
    print("  • Q2_goal_rate_vs_angle.png")
    print("  • Q3_empty_net_goals_histogram.png")


if __name__ == "__main__":
    generate_all_visualizations()
