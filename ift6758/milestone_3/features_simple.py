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
sys.path.insert(0, str(Path(__file__).parent))
from pandas_conversion import get_seasons_dataframe, decode_situation_code


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


def _calculate_powerplay_features(df: pd.DataFrame) -> pd.DataFrame: # not used in this script bc we only generate simple features
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
    project_root = Path(__file__).parent
    
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


def detect_empty_net_from_situation(df):
    """
    Determine empty-net goals using decoded goalie presence, using the full
    home/away → friendly/opponent interpretation.

    - empty net = opponent_goalie == 0
    - NOT empty net = friendly_goalie == 0 (goalie pulled on shooter's team)
    """

    # --- If goalie columns missing, recompute using situationCode ---
    missing = ("friendly_goalie" not in df.columns) or ("opponent_goalie" not in df.columns)

    if missing:
        if "situationCode" not in df.columns or "teamId" not in df.columns:
            raise ValueError(
                "Cannot compute empty-net status: situationCode or teamId missing.")

    # --- Compute empty net ---
    df["empty_net"] = 0
    df.loc[df["opponent_goalie"] == 0, "empty_net"] = 1
    df.loc[df["friendly_goalie"] == 0, "empty_net"] = 0

    return df

def clean_dataframe(begin: int, end: int) -> pd.DataFrame:
    """
    Charge et nettoie les données d'une plage de saisons.
    Conserve à la fois :
        - les variables nécessaires au tableau de bord Streamlit
        - les 4 features du modèle :
            distance_net, angle_net, is_goal, empty_net
    """

    logger.info(f"Chargement des saisons {begin}-{end}...")
    df = get_seasons_dataframe(begin, end).copy()

    # Garder uniquement tirs et buts
    df = df[df["typeDescKey"].isin(["shot-on-goal", "goal"])].copy()
    df["is_goal"] = (df["typeDescKey"] == "goal").astype(int)

    # ------------------------------------------------------------------
    # 1. Detect empty net using situationCode and friendly/opponent goalies
    # ------------------------------------------------------------------
    df = detect_empty_net_from_situation(df)

    # ------------------------------------------------------------------
    # 2. Compute distance and angle features
    # ------------------------------------------------------------------
    goal_x = np.where(df["xCoord"] > 0, 89, -89)
    goal_y = 0

    df["distance_net"] = np.sqrt(
        (goal_x - df["xCoord"]) ** 2 + (goal_y - df["yCoord"]) ** 2
    )

    df["angle_net"] = np.degrees(
        np.arctan2(np.abs(df["yCoord"]), np.abs(goal_x - df["xCoord"]))
    )

    # ------------------------------------------------------------------
    # 3. Select STREAMLIT dashboard-required variables
    # ------------------------------------------------------------------
    dashboard_fields = [
        "idGame",
        "teamAbbr",
        "teamId",
        "period",
        "timeInPeriod",
        "shotType",
        "xCoord",
        "yCoord",
        "shooterId",
        "shooterName",
        "goalieId",
        "goalieName",
        "typeDescKey",
        "zoneCode",
    ]

    # Ensure fields exist (avoid KeyErrors for older seasons)
    available_fields = [col for col in dashboard_fields if col in df.columns]

    # ------------------------------------------------------------------
    # 4. Add the engineered features needed for xG model
    # ------------------------------------------------------------------
    model_fields = [
        "distance_net",
        "angle_net",
        "empty_net",
        "is_goal",
    ]

    df_clean = df[available_fields + model_fields].copy()

    logger.info(
        f"DataFrame nettoyé: {len(df_clean)} tirs, "
        f"{len(model_fields)-1} features du modèle + {len(available_fields)} champs dashboard + 1 champ prédictif"
    )

    return df_clean



def generate_train_val_test_datasets(
    train_begin: int = 2016,
    train_end: int = 2020,  
    test_begin: int = 2020,
    test_end: int = 2021,   
    test_size: float = 0.2,
    random_state: int = 42,
    output_dir: str = None
):

    # ===============================
    #  Directory configuration
    # ===============================
    if output_dir is None:
        project_root = Path(__file__).parent
        output_dir = project_root / "data" / "processed"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Génération datasets (4 features) ===")

    # =======================================================
    #  1. Load TRAIN + VAL (2016–2020 → seasons 16–17 to 19–20)
    # =======================================================
    logger.info(f"Chargement TRAIN+VAL saisons {train_begin}-{train_end}...")
    df_train_val = clean_dataframe(train_begin, train_end)
    df_train_val = df_train_val.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # =======================================================
    #  2. Stratified 80/20 split
    # =======================================================
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=test_size,
        random_state=random_state,
        stratify=df_train_val["is_goal"]
    )

    # =======================================================
    #  3. Load TEST season (2020–2021)
    # =======================================================
    logger.info("Chargement TEST saison 2020–2021...")
    df_test_all = clean_dataframe(test_begin, test_end)

    # ======================
    # 3A. Regular-season test
    # ======================
    df_test_regular = df_test_all[df_test_all["idGame"].astype(str).str[4:6] != "03"].copy()

    # ======================
    # 3B. Playoff test
    # ======================
    df_test_playoffs = df_test_all[df_test_all["idGame"].astype(str).str[4:6] == "03"].copy()

    # =======================================================
    #  4. Save outputs
    # =======================================================
    train_path = output_dir / "train_data.csv"
    val_path = output_dir / "val_data.csv"
    test_regular_path = output_dir / "test_regular.csv"
    test_playoffs_path = output_dir / "test_playoffs.csv"

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test_regular.to_csv(test_regular_path, index=False)
    df_test_playoffs.to_csv(test_playoffs_path, index=False)

    # =======================================================
    #  Print summary
    # =======================================================
    logger.info("=== Fini ===")
    logger.info(f"Train:         {len(df_train)} rows → {train_path}")
    logger.info(f"Validation:    {len(df_val)} rows → {val_path}")
    logger.info(f"Test regular:  {len(df_test_regular)} rows → {test_regular_path}")
    logger.info(f"Test playoffs: {len(df_test_playoffs)} rows → {test_playoffs_path}")

    return df_train, df_val, df_test_regular, df_test_playoffs

def main():
    """Fonction principale pour générer les datasets."""
    generate_train_val_test_datasets()


if __name__ == "__main__":
    main()