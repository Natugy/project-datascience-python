"""
Conversion des données JSON NHL en DataFrames pandas.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

# Imports robustes pour différents contextes d'exécution
try:
    from .data_scrapping import LNHDataScrapper
except ImportError:
    try:
        from ift6758.data.data_scrapping import LNHDataScrapper
    except ImportError:
        from data_scrapping import LNHDataScrapper


def get_playerName_from_game(game_players_data: pd.DataFrame, player_id) -> Optional[str]:
    """
    Récupère le nom complet d'un joueur à partir de son ID.
    
    Args:
        game_players_data: DataFrame des joueurs du match
        player_id: ID du joueur recherché
        
    Returns:
        Nom complet du joueur ou None si introuvable
    """
    if pd.isna(player_id) or game_players_data.empty:
        return None
    
    player = game_players_data.loc[game_players_data['playerId'] == player_id]
    
    if len(player) == 0:
        return None
    
    first_name = player['firstName.default'].iloc[0]
    last_name = player['lastName.default'].iloc[0]
    return f"{first_name} {last_name}"


# Colonnes standardisées pour tous les types d'événements
STANDARD_COLUMNS = [
    'teamId', 'period', 'timeInPeriod', 'shotType', 'xCoord', 'yCoord',
    'shooterId', 'shooterName', 'goalieId', 'goalieName', 'typeDescKey', 'zoneCode'
]


def _extract_event_dataframe(
    players: pd.DataFrame,
    plays: List[Dict],
    event_types: List[str],
    shooter_id_field: str = "shootingPlayerId",
    scorer_id_field: str = "scoringPlayerId"
) -> pd.DataFrame:
    """
    Fonction générique pour extraire un DataFrame d'événements.
    Élimine la duplication de code entre shot-on-goal, goal, missed-shot.
    
    Args:
        players: DataFrame des joueurs du match
        plays: Liste des événements du match
        event_types: Types d'événements à filtrer (ex: ['goal'])
        shooter_id_field: Nom du champ pour l'ID du tireur dans details
        scorer_id_field: Nom du champ alternatif pour les buts
        
    Returns:
        DataFrame avec les colonnes standardisées
    """
    df = pd.DataFrame(plays)
    
    # Retourner DataFrame vide si pas de données
    if df.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    
    # Extraire les details et période
    details = pd.json_normalize(df["details"])
    period = pd.json_normalize(df['periodDescriptor'])
    
    # Filtrer par type d'événement
    df = df[df["typeDescKey"].isin(event_types)].copy()
    
    if df.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    
    # Extraire les coordonnées
    df["xCoord"] = details.get("xCoord")
    df["yCoord"] = details.get("yCoord")
    df["zoneCode"] = details.get("zoneCode")
    
    # Déterminer quel champ utiliser pour le tireur
    if "goal" in event_types:
        df["shooterId"] = details.get(scorer_id_field)
    else:
        df["shooterId"] = details.get(shooter_id_field)
    
    df["shooterName"] = df["shooterId"].apply(
        lambda pid: get_playerName_from_game(players, pid)
    )
    
    # Gardien et type de tir
    df["goalieId"] = details.get("goalieInNetId")
    df["goalieName"] = df["goalieId"].apply(
        lambda pid: get_playerName_from_game(players, pid)
    )
    df["shotType"] = details.get("shotType")
    df["period"] = period.get("number")
    df["teamId"] = details.get("eventOwnerTeamId")
    
    return df[STANDARD_COLUMNS]


def get_dataframe_from_shot_on_goal_event(players: pd.DataFrame, plays: List[Dict]) -> pd.DataFrame:
    """Extrait les événements de tirs au but (shot-on-goal)."""
    return _extract_event_dataframe(
        players, plays, 
        event_types=['shot-on-goal'],
        shooter_id_field="shootingPlayerId"
    )


def get_dataframe_from_goal_event(players: pd.DataFrame, plays: List[Dict]) -> pd.DataFrame:
    """Extrait les événements de buts."""
    return _extract_event_dataframe(
        players, plays,
        event_types=['goal'],
        scorer_id_field="scoringPlayerId"
    )


def get_dataframe_from_missed_shot_event(players: pd.DataFrame, plays: List[Dict]) -> pd.DataFrame:
    """Extrait les événements de tirs manqués."""
    return _extract_event_dataframe(
        players, plays,
        event_types=['missed-shot'],
        shooter_id_field="shootingPlayerId"
    )


def get_dataframe_from_other_event(players: pd.DataFrame, plays: List[Dict]) -> pd.DataFrame:
    """
    Extrait les autres événements (non-tirs).
    Utile pour le contexte (faceoff, hit, etc.)
    """
    df = pd.DataFrame(plays)
    
    # Événements de tirs à exclure
    shot_events = ['missed-shot', 'goal', 'shot-on-goal']
    
    if df.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    
    details = pd.json_normalize(df["details"])
    period = pd.json_normalize(df['periodDescriptor'])
    
    # Garder uniquement les non-tirs
    df = df[~df["typeDescKey"].isin(shot_events)].copy()
    
    if df.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    
    # Pour les autres événements, pas de données de tir
    df["xCoord"] = details.get("xCoord")
    df["yCoord"] = details.get("yCoord")
    df["zoneCode"] = details.get("zoneCode")
    df["shooterId"] = None
    df["shooterName"] = None
    df["goalieId"] = None
    df["goalieName"] = None
    df["shotType"] = None
    df["period"] = period.get("number")
    df["teamId"] = details.get("eventOwnerTeamId")
    
    return df[STANDARD_COLUMNS]


def _game_team_meta(game: Dict) -> Dict:
    """
    Extrait les métadonnées des équipes (home/away) d'un match.
    
    Returns:
        Dict avec home_id, home_abbr, away_id, away_abbr
    """
    home = game.get("homeTeam", {}) or game.get("home", {})
    away = game.get("awayTeam", {}) or game.get("away", {})
    
    return {
        "home_id": home.get("id"),
        "home_abbr": home.get("abbrev") or home.get("triCode"),
        "away_id": away.get("id"),
        "away_abbr": away.get("abbrev") or away.get("triCode"),
    }


def _owner_to_abbr(owner_id, meta: Dict) -> Optional[str]:
    """Convertit un teamId en abréviation (ex: 8 → 'MTL')."""
    if owner_id == meta.get("home_id"):
        return meta.get("home_abbr")
    if owner_id == meta.get("away_id"):
        return meta.get("away_abbr")
    return None


def get_dataframe_from_data(season: str) -> pd.DataFrame:
    """
    Convertit toutes les données JSON d'une saison en DataFrame pandas.
    Sauvegarde automatiquement en CSV pour cache.
    
    Args:
        season: Saison au format "20162017"
        
    Returns:
        DataFrame avec tous les événements de la saison
    """
    scrapper = LNHDataScrapper()
    
    # Trouver la racine du projet
    project_root = Path(__file__).parent.parent.parent
    dest_folder = project_root / "data" / "raw"
    dest_folder.mkdir(parents=True, exist_ok=True)
    
    # Utiliser data/raw/ pour les CSV
    data_csv = dest_folder / f"{season}.csv"
    
    # Si le CSV existe déjà, le charger directement
    if data_csv.exists():
        print(f"Chargement du CSV existant: {data_csv}")
        return pd.read_csv(data_csv)
    
    # Charger les données JSON
    print(f"Conversion de la saison {season} en DataFrame...")
    games = scrapper.open_data(season)
    result = pd.DataFrame()
    
    for game in games:
        # Extraire les joueurs et événements
        players = pd.json_normalize(game.get("rosterSpots", []))
        plays = game.get("plays", [])
        
        # Extraire les différents types d'événements
        df_sog = get_dataframe_from_shot_on_goal_event(players, plays)
        df_goal = get_dataframe_from_goal_event(players, plays)
        df_miss = get_dataframe_from_missed_shot_event(players, plays)
        df_other = get_dataframe_from_other_event(players, plays)
        
        # Combiner (filtrer les vides pour éviter warnings)
        dfs = [df for df in [df_sog, df_goal, df_miss, df_other] if not df.empty]
        
        if not dfs:
            continue
        
        # Suppression du FutureWarning en spécifiant comment gérer les colonnes vides
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning)
            df_game = pd.concat(dfs, ignore_index=True)
        
        # Enrichir avec métadonnées du match
        game_id = game.get("id") or game.get("gamePk") or game.get("gameId")
        game_season = game.get("season")
        df_game["idGame"] = game_id
        df_game["season"] = game_season
        
        # Ajouter l'abréviation d'équipe
        meta = _game_team_meta(game)
        df_game["teamAbbr"] = df_game["teamId"].apply(
            lambda tid: _owner_to_abbr(tid, meta)
        )
        
        # Nettoyage minimal
        df_game = df_game.dropna(subset=["xCoord", "yCoord"])
        df_game = df_game.sort_values(by=['period', 'timeInPeriod'])
        
        result = pd.concat([result, df_game], ignore_index=True)
    
    # Sauvegarder le CSV dans data/raw/
    if not result.empty:
        print(f"Sauvegarde de {len(result)} événements dans {data_csv}")
        result.to_csv(data_csv, index=False)
    
    return result


def get_seasons_dataframe(begin: int, end: int) -> pd.DataFrame:
    """
    Combine les données de plusieurs saisons en un seul DataFrame.
    
    Args:
        begin: Année de début (ex: 2016)
        end: Année de fin (ex: 2020)
        
    Returns:
        DataFrame combiné de toutes les saisons
    """
    print(f"Chargement des saisons {begin}-{end}...")
    
    result = pd.DataFrame()
    seasons = [f"{y}{y+1}" for y in range(begin, end)]
    
    for season in seasons:
        df_season = get_dataframe_from_data(season)
        result = pd.concat([result, df_season], ignore_index=True)
        print(f"{season}: {len(df_season)} événements")

    print(f"Total: {len(result)} événements")
    return result


def main():
    """Fonction principale pour convertir toutes les saisons."""
    print("=== Conversion JSON → CSV ===")
    
    for season in [f"{y}{y+1}" for y in range(2016, 2024)]:
        df = get_dataframe_from_data(season)
        print(f"{season}: {len(df)} événements sauvegardés")

    print("=== Conversion terminée ===")


if __name__ == "__main__":
    main()
