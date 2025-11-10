"""
Feature engineering pour la prédiction de buts au hockey (xG).
Génère les datasets train/val/test avec toutes les features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Import simple
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.pandas_conversion import get_seasons_dataframe


def _convert_time_to_seconds(time_str) -> float:
    """
    Convertit une chaîne de temps MM:SS en secondes.
    
    Args:
        time_str: Temps au format "MM:SS"
        
    Returns:
        Nombre de secondes (float), ou NaN si invalide
    """
    try:
        minutes, seconds = map(int, str(time_str).split(":"))
        return minutes * 60 + seconds
    except (ValueError, AttributeError):
        return np.nan


def _calculate_powerplay_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les features de power-play pour chaque tir (BONUS 5%).
    
    Features calculées:
    - time_since_powerplay_start: Temps depuis début PP (secondes), 0 si pas en PP
    - friendly_skaters: Nombre de joueurs non-gardiens de l'équipe qui tire (3-5)
    - opponent_skaters: Nombre de joueurs non-gardiens adverses (3-5)
    
    Args:
        df: DataFrame avec les événements triés par match et temps
        
    Returns:
        DataFrame avec les 3 nouvelles colonnes
    """
    # Initialiser les colonnes
    df["time_since_powerplay_start"] = 0.0
    df["friendly_skaters"] = 5  # Par défaut: 5 joueurs (sans gardien)
    df["opponent_skaters"] = 5
    
    # Charger les événements bruts incluant les pénalités
    from data.pandas_conversion import LNHDataScrapper
    
    scrapper = LNHDataScrapper()
    project_root = Path(__file__).parent.parent.parent
    
    # Charger toutes les saisons nécessaires une seule fois
    seasons_to_load = {}
    for season in df['season'].dropna().unique():
        season_str = str(int(season))
        if season_str not in seasons_to_load:
            try:
                seasons_to_load[season_str] = {
                    g.get('id'): g for g in scrapper.open_data(season_str)
                }
            except Exception as e:
                logger.warning(f"Impossible de charger la saison {season_str}: {e}")
                seasons_to_load[season_str] = {}
    
    # Traiter chaque match individuellement
    for game_id in df['idGame'].unique():
        if pd.isna(game_id):
            continue
            
        game_id = int(game_id)
        season = df[df['idGame'] == game_id]['season'].iloc[0]
        season_str = str(int(season))
        
        try:
            # Récupérer les données du match depuis le cache
            game_data = seasons_to_load.get(season_str, {}).get(game_id)
            
            if not game_data:
                continue
            
            # Extraire tous les événements (incluant pénalités)
            plays = game_data.get('plays', [])
            
            # Construire une liste des pénalités avec leur timing
            penalties = []
            for play in plays:
                if play.get('typeDescKey') == 'penalty':
                    period = play['periodDescriptor']['number']
                    time_in_period = play['timeInPeriod']
                    
                    # Convertir en secondes absolues
                    time_sec = _convert_time_to_seconds(time_in_period)
                    game_sec = (period - 1) * 1200 + time_sec
                    
                    details = play.get('details', {})
                    duration_min = details.get('duration', 2)  # Par défaut 2 min
                    team_id = details.get('eventOwnerTeamId')
                    
                    # Ignorer les pénalités de match (duration >= 10 min)
                    if duration_min >= 10:
                        continue
                    
                    penalties.append({
                        'game_sec': game_sec,
                        'duration_sec': duration_min * 60,
                        'team_id': team_id,
                        'expires_at': game_sec + duration_min * 60
                    })
            
            # Trier les pénalités par temps
            penalties.sort(key=lambda x: x['game_sec'])
            
            # Appliquer les features PP aux tirs de ce match
            game_mask = df['idGame'] == game_id
            
            for idx in df[game_mask].index:
                shot_time = df.loc[idx, 'game_seconds']
                shot_team_id = df.loc[idx, 'teamId']
                
                # Trouver les pénalités actives au moment du tir
                active_penalties_friendly = []  # Pénalités contre l'équipe qui tire
                active_penalties_opponent = []  # Pénalités contre l'adversaire
                
                for pen in penalties:
                    # La pénalité est-elle active?
                    if pen['game_sec'] <= shot_time < pen['expires_at']:
                        if pen['team_id'] == shot_team_id:
                            # Pénalité contre l'équipe qui tire (désavantage)
                            active_penalties_friendly.append(pen)
                        else:
                            # Pénalité contre l'adversaire (avantage/PP)
                            active_penalties_opponent.append(pen)
                
                # Calculer le nombre de joueurs
                friendly_skaters = 5 - len(active_penalties_friendly)
                opponent_skaters = 5 - len(active_penalties_opponent)
                
                # Limiter entre 3 et 5
                friendly_skaters = max(3, min(5, friendly_skaters))
                opponent_skaters = max(3, min(5, opponent_skaters))
                
                df.loc[idx, 'friendly_skaters'] = friendly_skaters
                df.loc[idx, 'opponent_skaters'] = opponent_skaters
                
                # Calculer time_since_powerplay_start
                if len(active_penalties_opponent) > 0:
                    # En power-play (avantage numérique)
                    # Prendre la pénalité la plus ancienne encore active
                    oldest_pp = min(active_penalties_opponent, key=lambda x: x['game_sec'])
                    df.loc[idx, 'time_since_powerplay_start'] = shot_time - oldest_pp['game_sec']
                elif len(active_penalties_friendly) > 0:
                    # En désavantage numérique
                    df.loc[idx, 'time_since_powerplay_start'] = 0.0
                else:
                    # Situation à forces égales
                    df.loc[idx, 'time_since_powerplay_start'] = 0.0
        
        except Exception as e:
            # En cas d'erreur, garder les valeurs par défaut pour ce match
            logger.warning(f"Impossible de calculer les PP features pour le match {game_id}: {e}")
            continue
    
    return df


def clean_dataframe(begin: int, end: int) -> pd.DataFrame:
    """
    Charge et nettoie les données d'une plage de saisons.
    Génère toutes les features pour la prédiction de buts.
    
    Args:
        begin: Année de début (ex: 2016)
        end: Année de fin (ex: 2020)
        
    Returns:
        DataFrame nettoyé avec toutes les features
    """
    logger.info(f"Chargement des saisons {begin}-{end}...")
    df = get_seasons_dataframe(begin, end)
    df = df.copy()
    
    # Trier par match et temps
    logger.info("Tri des événements par match et période...")
    df = df.sort_values(by=["idGame", "period", "timeInPeriod"]).reset_index(drop=True)
    
    # === Features de contexte: événement précédent ===
    logger.info("Calcul des features d'événement précédent...")
    df["prev_event"] = df.groupby("idGame")["typeDescKey"].shift(1)
    df["prev_team"] = df.groupby("idGame")["teamAbbr"].shift(1)
    df["prev_xCoord"] = df.groupby("idGame")["xCoord"].shift(1)
    df["prev_yCoord"] = df.groupby("idGame")["yCoord"].shift(1)
    df["prev_time"] = df.groupby("idGame")["timeInPeriod"].shift(1)
    
    # === Filtrage: garder uniquement tirs et buts ===
    logger.info("Filtrage des tirs et buts...")
    df = df[df["typeDescKey"].isin(["shot-on-goal", "goal"])].copy()
    df["is_goal"] = (df["typeDescKey"] == "goal").astype(int)
    
    logger.info(f"  {len(df)} tirs/buts retenus ({df['is_goal'].sum()} buts)")
    
    # === Feature: Empty net ===
    df["empty_net"] = df.get("emptyNet", pd.Series(0, index=df.index)).fillna(0).astype(int)
    
    # === Features géométriques: distance et angle au filet ===
    logger.info("Calcul de distance_net et angle_net...")
    
    # Le filet est à x=±89 pieds (selon le côté de la glace)
    goal_x = np.where(df["xCoord"] > 0, 89, -89)
    goal_y = 0
    
    df["distance_net"] = np.sqrt(
        (goal_x - df["xCoord"])**2 + (goal_y - df["yCoord"])**2
    )
    
    df["angle_net"] = np.degrees(
        np.arctan2(np.abs(df["yCoord"]), np.abs(goal_x - df["xCoord"]))
    )
    
    # === Feature: Période de jeu ===
    df["game_period"] = df["period"]
    
    # === Feature: Temps de jeu (game_seconds) ===
    logger.info("Calcul de game_seconds...")
    
    # Convertir timeInPeriod en secondes
    df["time_sec"] = df["timeInPeriod"].apply(_convert_time_to_seconds)
    df["prev_time_sec"] = df["prev_time"].apply(_convert_time_to_seconds)
    
    # Game seconds = (période - 1) * 1200 + secondes dans la période
    # Périodes 1-3: 20 min = 1200s chacune
    df["game_seconds"] = (df["period"] - 1) * 1200 + df["time_sec"]
    
    # === Feature: Type de tir ===
    df["shot_type"] = df.get("shotType", pd.Series("Unknown", index=df.index)).fillna("Unknown")
    
    # === Features avancées: Rebond ===
    logger.info("Calcul des features de rebond...")
    
    df["is_rebound"] = df["prev_event"].isin(["shot-on-goal", "goal"]).astype(int)
    
    # Angle de l'événement précédent
    prev_goal_x = np.where(df["prev_xCoord"] > 0, 89, -89)
    df["prev_angle_net"] = np.degrees(
        np.arctan2(np.abs(df["prev_yCoord"]), np.abs(prev_goal_x - df["prev_xCoord"]))
    )
    
    # Changement d'angle (uniquement si rebond)
    df["change_in_angle"] = np.where(
        df["is_rebound"] == 1,
        np.abs(df["angle_net"] - df["prev_angle_net"]),
        0
    )
    
    # === Features avancées: Vitesse et distance ===
    logger.info("Calcul de shot_speed et distance_prev_event...")
    
    # Delta temps entre événements
    df["delta_t"] = df["time_sec"] - df["prev_time_sec"]
    
    # Distance depuis événement précédent
    df["distance_prev_event"] = np.sqrt(
        (df["xCoord"] - df["prev_xCoord"])**2 + 
        (df["yCoord"] - df["prev_yCoord"])**2
    )
    
    # Vitesse du tir (distance / temps)
    df["shot_speed"] = np.where(
        df["delta_t"] > 0,
        df["distance_prev_event"] / df["delta_t"],
        0
    )
    
    # === BONUS: Features de Power-Play ===
    logger.info("Calcul des features de power-play (BONUS)...")
    df = _calculate_powerplay_features(df)
    
    # === Sélection des colonnes finales ===
    logger.info("Sélection des features finales...")
    
    df_clean = df[[
        # Coordonnées actuelles
        "xCoord", "yCoord",
        # Géométrie
        "distance_net", "angle_net",
        # Target et empty net
        "is_goal", "empty_net",
        # Temps
        "game_seconds", "game_period",
        # Type de tir
        "shot_type",
        # Coordonnées événement précédent (REQUIS par le devoir)
        "prev_xCoord", "prev_yCoord",
        # Temps écoulé depuis événement précédent (REQUIS par le devoir)
        "delta_t",
        # Features dérivées
        "is_rebound", "change_in_angle", "shot_speed", "distance_prev_event",
        # Power-play features (BONUS 5%)
        "time_since_powerplay_start", "friendly_skaters", "opponent_skaters",
        # Métadonnées
        "season", "teamAbbr", "idGame",
        # Contexte
        "prev_event", "prev_team"
    ]].copy()
    
    logger.info(f"DataFrame nettoyé: {len(df_clean)} tirs, {len(df_clean.columns)} features")
    logger.info(f"  Features ajoutées: prev_xCoord, prev_yCoord, delta_t")
    
    return df_clean


def figure_ratio_but_nonbut(df_clean: pd.DataFrame) -> None:
    """
    Génère des graphiques du taux de buts en fonction de la distance et de l'angle.
    Utile pour l'analyse exploratoire.
    
    Args:
        df_clean: DataFrame nettoyé avec features
    """
    logger.info("Génération des graphiques de taux de but...")
    
    # Conversion en numérique (au cas où)
    df_clean["distance_net"] = pd.to_numeric(df_clean["distance_net"], errors="coerce")
    df_clean["angle_net"] = pd.to_numeric(df_clean["angle_net"], errors="coerce")
    
    # Bins pour distance et angle
    bin_distance = np.arange(0, 90, 5)
    bin_angle = np.arange(0, 90, 5)
    
    # === Graphique 1: Taux de but par distance ===
    df_clean["distance_bin"] = pd.cut(df_clean["distance_net"], bins=bin_distance)
    
    goal_rate_distance = (
        df_clean.groupby("distance_bin", observed=True)["is_goal"]
        .mean()
        .reset_index()
        .rename(columns={"is_goal": "goal_rate"})
    )
    
    goal_rate_distance["distance_mid"] = goal_rate_distance["distance_bin"].apply(
        lambda x: x.mid if pd.notnull(x) else np.nan
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(goal_rate_distance["distance_mid"], goal_rate_distance["goal_rate"], 
             marker="o", linewidth=2)
    plt.title("Taux de but en fonction de la distance au filet", fontsize=14, fontweight='bold')
    plt.xlabel("Distance (pieds)", fontsize=12)
    plt.ylabel("Taux de but (#buts / #tirs)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # === Graphique 2: Taux de but par angle ===
    df_clean["angle_bin"] = pd.cut(df_clean["angle_net"], bins=bin_angle)
    
    goal_rate_angle = (
        df_clean.groupby("angle_bin", observed=True)["is_goal"]
        .mean()
        .reset_index()
        .rename(columns={"is_goal": "goal_rate"})
    )
    
    goal_rate_angle["angle_mid"] = goal_rate_angle["angle_bin"].apply(
        lambda x: x.mid if pd.notnull(x) else np.nan
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(goal_rate_angle["angle_mid"], goal_rate_angle["goal_rate"], 
             marker="o", color="orange", linewidth=2)
    plt.title("Taux de but en fonction de l'angle du tir", fontsize=14, fontweight='bold')
    plt.xlabel("Angle (degrés)", fontsize=12)
    plt.ylabel("Taux de but (#buts / #tirs)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def generate_train_val_test_datasets(
    train_begin: int = 2016,
    train_end: int = 2020,
    test_begin: int = 2020,
    test_end: int = 2021,
    test_size: float = 0.2,
    random_state: int = 42,
    output_dir: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Génère les ensembles train/validation/test pour l'entraînement des modèles.
    Sauvegarde automatiquement dans data/processed/.
    
    Args:
        train_begin: Année de début pour train+val (défaut: 2016)
        train_end: Année de fin pour train+val (défaut: 2020)
        test_begin: Année de début pour test (défaut: 2020)
        test_end: Année de fin pour test (défaut: 2021)
        test_size: Proportion de validation (défaut: 0.2 = 20%)
        random_state: Seed pour reproductibilité (défaut: 42)
        output_dir: Dossier de sortie (défaut: data/processed/)
        
    Returns:
        Tuple (df_train, df_val, df_test)
    """
    # Déterminer le dossier de sortie
    if output_dir is None:
        project_root = Path(__file__).parent.parent.parent
        output_dir = project_root / "data" / "processed"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=== Génération des datasets train/val/test ===")
    
    # Charger et nettoyer les données de train+validation
    logger.info(f"Chargement des donnees {train_begin}/{train_begin+1} - {train_end-1}/{train_end} (train + validation)")
    df_train_val = clean_dataframe(train_begin, train_end)
    
    # Mélanger pour éviter les biais temporels
    logger.info("Melange des donnees...")
    df_train_val = df_train_val.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Charger les données de test
    logger.info(f"Chargement des donnees {test_begin}/{test_begin+1} - {test_end-1}/{test_end} (test)")
    df_test = clean_dataframe(test_begin, test_end)
    
    # Split train/validation avec stratification
    logger.info(f"Split train/validation ({int((1-test_size)*100)}/{int(test_size*100)})...")
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=test_size,
        random_state=random_state,
        stratify=df_train_val['is_goal']
    )
    
    # Sauvegarder les ensembles
    logger.info(f"Sauvegarde des datasets dans {output_dir}...")
    
    train_path = output_dir / "train_data.csv"
    val_path = output_dir / "val_data.csv"
    test_path = output_dir / "test_data.csv"
    
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)
    
    # Résumé
    logger.info("=== Ensembles de donnees crees ===")
    logger.info(f"Train: {len(df_train):,} tirs, {df_train['is_goal'].sum():,} buts ({df_train['is_goal'].mean()*100:.2f}%)")
    logger.info(f"Val:   {len(df_val):,} tirs, {df_val['is_goal'].sum():,} buts ({df_val['is_goal'].mean()*100:.2f}%)")
    logger.info(f"Test:  {len(df_test):,} tirs, {df_test['is_goal'].sum():,} buts ({df_test['is_goal'].mean()*100:.2f}%)")
    logger.info(f"Fichiers sauvegardes:")
    logger.info(f"  - {train_path}")
    logger.info(f"  - {val_path}")
    logger.info(f"  - {test_path}")
    
    return df_train, df_val, df_test


def main():
    """Fonction principale pour générer les datasets."""
    generate_train_val_test_datasets()


if __name__ == "__main__":
    main()

