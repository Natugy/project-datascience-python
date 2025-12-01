"""
Converts NHL JSON files scrapped from API to pandas DataFrames
"""
# ============================================================
#              IMPORTS AND BASIC CONFIGURATIONS
# ============================================================
# basic python
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging

# robust imports for different utilisation contexts
try:
    from .data_scrapping_correct import LNHDataScrapper
except ImportError:
    try:
        from milestone_3.data_scrapping_correct import LNHDataScrapper
    except ImportError:
        from data_scrapping_correct import LNHDataScrapper


# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
#                  UTILITY FUNCTIONS
# ============================================================

def get_playerName_from_game(game_players_data: pd.DataFrame, player_id) -> Optional[str]:
    """Return player's full name or None."""
    if pd.isna(player_id) or game_players_data.empty:
        return None
    
    player = game_players_data.loc[game_players_data['playerId'] == player_id]
    if len(player) == 0:
        return None
    
    return f"{player['firstName.default'].iloc[0]} {player['lastName.default'].iloc[0]}"


def _game_team_meta(game: Dict) -> Dict:
    """Extract home/away IDs and abbreviations."""
    home = game.get("homeTeam", {}) or game.get("home", {})
    away = game.get("awayTeam", {}) or game.get("away", {})

    return {
        "home_id": home.get("id"),
        "home_abbr": home.get("abbrev") or home.get("triCode"),
        "away_id": away.get("id"),
        "away_abbr": away.get("abbrev") or away.get("triCode"),
    }


def _owner_to_abbr(owner_id, meta: Dict) -> Optional[str]:
    if owner_id == meta["home_id"]:
        return meta["home_abbr"]
    if owner_id == meta["away_id"]:
        return meta["away_abbr"]
    return None


# Standard output columns
STANDARD_COLUMNS = [
    'teamId', 'period', 'timeInPeriod', 'shotType', 'xCoord', 'yCoord',
    'shooterId', 'shooterName', 'goalieId', 'goalieName', 'typeDescKey', 'zoneCode',
    'situationCode', 'friendly_skaters', 'opponent_skaters',
    'friendly_goalie', 'opponent_goalie'
]

#=====================
# NHL SITUATION CODE DECODING
#=====================
def decode_situation_code(code, event_team_id, home_id, away_id):
    """
    Decode situationCode based on NHL documentation.

    Digits = AWAY → HOME:
        code[0] = away goalie present (1/0)
        code[1] = away skaters including goalie
        code[2] = home skaters including goalie
        code[3] = home goalie present (1/0)

    Convert to FRIENDLY vs OPPONENT using eventOwnerTeamId.
    """
    if pd.isna(code):
        return 5, 5, 1, 1

    code = str(int(code))
    if len(code) != 4 or not code.isdigit():
        return 5, 5, 1, 1

    away_goalie = int(code[0])
    away_total  = int(code[1])
    home_total  = int(code[2])
    home_goalie = int(code[3])

    away_skaters = away_total - away_goalie
    home_skaters = home_total - home_goalie

    shooter_is_home = (event_team_id == home_id)
    shooter_is_away = (event_team_id == away_id)

    if shooter_is_home:
        return home_skaters, away_skaters, home_goalie, away_goalie
    elif shooter_is_away:
        return away_skaters, home_skaters, away_goalie, home_goalie
    else:
        return 5, 5, 1, 1


# ============================================================
#                 CORE EXTRACTION FUNCTION
# ============================================================

def _extract_event_dataframe(
    players: pd.DataFrame,
    plays: List[Dict],
    event_types: List[str],
    home_id: int,
    away_id: int,
    shooter_id_field: str = "shootingPlayerId",
    scorer_id_field: str = "scoringPlayerId"
) -> pd.DataFrame:

    df = pd.DataFrame(plays)
    if df.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    details = pd.json_normalize(df["details"])
    period  = pd.json_normalize(df["periodDescriptor"])

    # situationCode from root or fallback
    root_sc = df.get("situationCode")
    details_sc = details.get("situationCode")
    df["situationCode"] = root_sc if root_sc is not None else details_sc

    # Filter by event type EARLY
    df = df[df["typeDescKey"].isin(event_types)].copy()
    if df.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    # Basic fields
    df["xCoord"]   = details.get("xCoord")
    df["yCoord"]   = details.get("yCoord")
    df["zoneCode"] = details.get("zoneCode")
    df["teamId"]   = details.get("eventOwnerTeamId")

    # Shooter ID (goal vs shot)
    if "goal" in event_types:
        df["shooterId"] = details.get(scorer_id_field)
    else:
        df["shooterId"] = details.get(shooter_id_field)

    df["shooterName"] = df["shooterId"].apply(
        lambda pid: get_playerName_from_game(players, pid)
    )

    # Goalie
    df["goalieId"] = details.get("goalieInNetId")
    df["goalieName"] = df["goalieId"].apply(
        lambda pid: get_playerName_from_game(players, pid)
    )

    df["shotType"] = details.get("shotType")
    df["period"]   = period.get("number")
    df["timeInPeriod"] = df["timeInPeriod"] if "timeInPeriod" in df else details.get("timeInPeriod")

    # decoding situation code
    decoded = df.apply(
        lambda row: decode_situation_code(
            code=row["situationCode"],
            event_team_id=row["teamId"],
            home_id=home_id,
            away_id=away_id
        ),
        axis=1
    )

    df["friendly_skaters"] = decoded.apply(lambda x: x[0]).astype(int)
    df["opponent_skaters"] = decoded.apply(lambda x: x[1]).astype(int)
    df["friendly_goalie"]  = decoded.apply(lambda x: x[2]).astype(int)
    df["opponent_goalie"]  = decoded.apply(lambda x: x[3]).astype(int)

    return df[STANDARD_COLUMNS]


# ============================================================
#                WRAPPER FUNCTIONS
# ============================================================

def get_dataframe_from_shot_on_goal_event(players, plays, home_id, away_id):
    return _extract_event_dataframe(
        players, plays,
        event_types=['shot-on-goal'],
        home_id=home_id,
        away_id=away_id,
        shooter_id_field="shootingPlayerId"
    )

def get_dataframe_from_goal_event(players, plays, home_id, away_id):
    return _extract_event_dataframe(
        players, plays,
        event_types=['goal'],
        home_id=home_id,
        away_id=away_id,
        scorer_id_field="scoringPlayerId"
    )


def get_dataframe_from_missed_shot_event(players, plays, home_id, away_id):
    return _extract_event_dataframe(
        players, plays,
        event_types=['missed-shot'],
        home_id=home_id,
        away_id=away_id,
        shooter_id_field="shootingPlayerId"
    )


def get_dataframe_from_other_event(players: pd.DataFrame, plays: List[Dict], home_id, away_id) -> pd.DataFrame:
    df = pd.DataFrame(plays)

    shot_events = ['missed-shot', 'goal', 'shot-on-goal']
    if df.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    details = pd.json_normalize(df["details"])
    period  = pd.json_normalize(df['periodDescriptor'])

    df = df[~df["typeDescKey"].isin(shot_events)].copy()
    if df.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    # Situation code
    root_sc    = df.get("situationCode")
    details_sc = details.get("situationCode")
    df["situationCode"] = root_sc if root_sc is not None else details_sc

    # set teamId
    df["teamId"] = details.get("eventOwnerTeamId")

    # Decode
    decoded = df.apply(
        lambda row: decode_situation_code(
            row["situationCode"],
            row["teamId"],
            home_id,
            away_id
        ),
        axis=1
    )

    df["friendly_skaters"] = decoded.apply(lambda x: x[0]).fillna(5).astype(int)
    df["opponent_skaters"] = decoded.apply(lambda x: x[1]).fillna(5).astype(int)
    df["friendly_goalie"]  = decoded.apply(lambda x: x[2]).fillna(1).astype(int)
    df["opponent_goalie"]  = decoded.apply(lambda x: x[3]).fillna(1).astype(int)

    # Now fill remaining fields
    df["xCoord"]      = details.get("xCoord")
    df["yCoord"]      = details.get("yCoord")
    df["zoneCode"]    = details.get("zoneCode")
    df["shooterId"]   = None
    df["shooterName"] = None
    df["goalieId"]    = None
    df["goalieName"]  = None
    df["shotType"]    = None
    df["period"]      = period.get("number")

    return df[STANDARD_COLUMNS]


# ============================================================
#             BUILD SEASON-LEVEL DATAFRAME
# ============================================================

def get_dataframe_from_data(season: str) -> pd.DataFrame:
    scrapper = LNHDataScrapper()

    project_root = Path(__file__).parent
    dest_folder = project_root / "data" / "raw"
    dest_folder.mkdir(parents=True, exist_ok=True)

    data_csv = dest_folder / f"{season}.csv"

    if data_csv.exists():
        print(f"Chargement du CSV existant: {data_csv}")
        return pd.read_csv(data_csv)

    print(f"Conversion de la saison {season} en DataFrame...")
    games = scrapper.open_data(season)
    result = pd.DataFrame()

    for game in games:
        players = pd.json_normalize(game.get("rosterSpots", []))
        plays   = game.get("plays", [])

        meta = _game_team_meta(game)
        home_id = meta["home_id"]
        away_id = meta["away_id"]

        df_sog = get_dataframe_from_shot_on_goal_event(players, plays, home_id, away_id)
        df_goal = get_dataframe_from_goal_event(players, plays, home_id, away_id)
        df_miss = get_dataframe_from_missed_shot_event(players, plays, home_id, away_id)
        df_other = get_dataframe_from_other_event(players, plays, home_id, away_id)

        dfs = [df for df in [df_sog, df_goal, df_miss, df_other]
            if df is not None and not df.empty and df.columns.size > 0]
        if not dfs:
            continue

        df_game = pd.concat(dfs, ignore_index=True)

        game_id = game.get("id") or game.get("gamePk") or game.get("gameId")
        season_id = game.get("season")

        df_game["idGame"] = game_id
        df_game["season"] = season_id
        df_game["teamAbbr"] = df_game["teamId"].apply(lambda tid: _owner_to_abbr(tid, meta))

        df_game = df_game.dropna(subset=["xCoord", "yCoord"])
        df_game = df_game.sort_values(by=["period", "timeInPeriod"])

        result = pd.concat([result, df_game], ignore_index=True)

    if not result.empty:
        print(f"Sauvegarde de {len(result)} événements dans {data_csv}")
        result.to_csv(data_csv, index=False)

    return result


def get_seasons_dataframe(begin: int, end: int) -> pd.DataFrame:
    print(f"Chargement des saisons {begin}-{end}...")
    result = pd.DataFrame()

    for y in range(begin, end):
        season = f"{y}{y+1}"
        df_season = get_dataframe_from_data(season)
        print(f"{season}: {len(df_season)} événements")
        result = pd.concat([result, df_season], ignore_index=True)

    print(f"Total: {len(result)} événements")
    return result


# ============================================================
#                    MAIN
# ============================================================

def main():
    print("=== Conversion JSON → CSV ===")
    for y in range(2016, 2024):
        season = f"{y}{y+1}"
        df = get_dataframe_from_data(season)
        print(f"{season}: {len(df)} événements sauvegardés")
    print("=== Conversion terminée ===")


if __name__ == "__main__":
    main()