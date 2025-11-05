import pandas as pd
import os

# --- imports robustes pour marcher en package, script, ou notebook ---
try:
    # Cas normal: import relatif quand appelé via "ift6758.data.pandas_conversion"
    from .data_scrapping import LNHDataScrapper
except Exception:
    try:
        # Cas où on est importé en absolu depuis la racine du repo
        from ift6758.data.data_scrapping import LNHDataScrapper
    except Exception:
        # Dernier recours (ex: exécution du .py directement, notebook dans le même dossier)
        from data_scrapping import LNHDataScrapper


def get_playerName_from_game(game_players_data: pd.DataFrame, searchedPlayerId):
    """Permet de récupérer le nom d'un joueur en donnant l'ID."""
    searchedPlayer = game_players_data.loc[game_players_data['playerId'] == searchedPlayerId]
    if len(searchedPlayer) == 0:
        return None
    first_name = searchedPlayer['firstName.default'].iloc[0]
    last_name = searchedPlayer['lastName.default'].iloc[0]
    return f"{first_name} {last_name}"


def get_dataframe_from_shot_on_goal_event(players, plays):
    df = pd.DataFrame(plays)
    filterValue = ['shot-on-goal']
    if df.empty:
        return pd.DataFrame(columns=[
            'teamId', 'period', 'timeInPeriod', 'shotType', 'xCoord', 'yCoord',
            'shooterId', 'shooterName', 'goalieId', 'goalieName', 'typeDescKey',
            'zoneCode'   
        ])
    details = pd.json_normalize(df["details"])
    period = pd.json_normalize(df['periodDescriptor'])
    df = df[df["typeDescKey"].isin(filterValue)].copy()
    if df.empty:
        return pd.DataFrame(columns=[
            'teamId', 'period', 'timeInPeriod', 'shotType', 'xCoord', 'yCoord',
            'shooterId', 'shooterName', 'goalieId', 'goalieName', 'typeDescKey',
            'zoneCode'
        ])
    df["xCoord"] = details.get("xCoord")
    df["yCoord"] = details.get("yCoord")
    df["zoneCode"] = details.get("zoneCode")             
    df["shooterId"] = details.get("shootingPlayerId")
    df["shooterName"] = df["shooterId"].apply(lambda pid: get_playerName_from_game(players, pid))
    df["goalieId"] = details.get("goalieInNetId")
    df["goalieName"] = df["goalieId"].apply(lambda pid: get_playerName_from_game(players, pid))
    df["shotType"] = details.get("shotType")
    df["period"] = period.get("number")
    df["teamId"] = details.get("eventOwnerTeamId")
    return df[['teamId', 'period', 'timeInPeriod', 'shotType', 'xCoord', 'yCoord',
               'shooterId', 'shooterName', 'goalieId', 'goalieName', 'typeDescKey',
               'zoneCode']]


def get_dataframe_from_goal_event(players, plays):
    df = pd.DataFrame(plays)
    filterValue = ['goal']
    if df.empty:
        return pd.DataFrame(columns=[
            'teamId', 'period', 'timeInPeriod', 'shotType', 'xCoord', 'yCoord',
            'shooterId', 'shooterName', 'goalieId', 'goalieName', 'typeDescKey',
            'zoneCode'
        ])
    details = pd.json_normalize(df["details"])
    period = pd.json_normalize(df['periodDescriptor'])
    df = df[df["typeDescKey"].isin(filterValue)].copy()
    if df.empty:
        return pd.DataFrame(columns=[
            'teamId', 'period', 'timeInPeriod', 'shotType', 'xCoord', 'yCoord',
            'shooterId', 'shooterName', 'goalieId', 'goalieName', 'typeDescKey',
            'zoneCode'
        ])
    df["xCoord"] = details.get("xCoord")
    df["yCoord"] = details.get("yCoord")
    df["zoneCode"] = details.get("zoneCode")           
    df["shooterId"] = details.get("scoringPlayerId")
    df["shooterName"] = df["shooterId"].apply(lambda pid: get_playerName_from_game(players, pid))
    df["goalieId"] = details.get("goalieInNetId")
    df["goalieName"] = df["goalieId"].apply(lambda pid: get_playerName_from_game(players, pid))
    df["shotType"] = details.get("shotType")
    df["period"] = period.get("number")
    df["teamId"] = details.get("eventOwnerTeamId")
    return df[['teamId', 'period', 'timeInPeriod', 'shotType', 'xCoord', 'yCoord',
               'shooterId', 'shooterName', 'goalieId', 'goalieName', 'typeDescKey',
               'zoneCode']]


def get_dataframe_from_missed_shot_event(players, plays):
    """inclut les tirs manqués pour avoir tous les tirs non bloqués."""
    df = pd.DataFrame(plays)
    filterValue = ['missed-shot']
    if df.empty:
        return pd.DataFrame(columns=[
            'teamId', 'period', 'timeInPeriod', 'shotType', 'xCoord', 'yCoord',
            'shooterId', 'shooterName', 'goalieId', 'goalieName', 'typeDescKey',
            'zoneCode'
        ])
    details = pd.json_normalize(df["details"])
    period = pd.json_normalize(df['periodDescriptor'])
    df = df[df["typeDescKey"].isin(filterValue)].copy()
    if df.empty:
        return pd.DataFrame(columns=[
            'teamId', 'period', 'timeInPeriod', 'shotType', 'xCoord', 'yCoord',
            'shooterId', 'shooterName', 'goalieId', 'goalieName', 'typeDescKey',
            'zoneCode'
        ])
    df["xCoord"] = details.get("xCoord")
    df["yCoord"] = details.get("yCoord")
    df["zoneCode"] = details.get("zoneCode")          
    # Pour missed-shot, pas toujours de shooterId, on tente 'shootingPlayerId'
    df["shooterId"] = details.get("shootingPlayerId")
    df["shooterName"] = df["shooterId"].apply(lambda pid: get_playerName_from_game(players, pid))
    df["goalieId"] = details.get("goalieInNetId")
    df["goalieName"] = df["goalieId"].apply(lambda pid: get_playerName_from_game(players, pid))
    df["shotType"] = details.get("shotType")
    df["period"] = period.get("number")
    df["teamId"] = details.get("eventOwnerTeamId")
    return df[['teamId', 'period', 'timeInPeriod', 'shotType', 'xCoord', 'yCoord',
               'shooterId', 'shooterName', 'goalieId', 'goalieName', 'typeDescKey',
               'zoneCode']]


def _game_team_meta(game: dict):
    """Retourne les IDs et abréviations home/away pour mapper teamAbbr."""
    H = game.get("homeTeam", {}) or game.get("home", {})
    A = game.get("awayTeam", {}) or game.get("away", {})
    return {
        "home_id": H.get("id"),
        "home_abbr": H.get("abbrev") or H.get("triCode"),
        "away_id": A.get("id"),
        "away_abbr": A.get("abbrev") or A.get("triCode"),
    }

def _owner_to_abbr(owner_id, meta):
    if owner_id == meta.get("home_id"):
        return meta.get("home_abbr")
    if owner_id == meta.get("away_id"):
        return meta.get("away_abbr")
    return None


def get_dataframe_from_other_event(players, plays):
    df = pd.DataFrame(plays)
    filterValue = ['missed-shot','goal','shot-on-goal']
    if df.empty:
        return pd.DataFrame(columns=[
            'teamId', 'period', 'timeInPeriod', 'shotType', 'xCoord', 'yCoord',
            'shooterId', 'shooterName', 'goalieId', 'goalieName', 'typeDescKey',
            'zoneCode'
        ])
    details = pd.json_normalize(df["details"])
    period = pd.json_normalize(df['periodDescriptor'])
    df = df[~df["typeDescKey"].isin(filterValue)].copy()
    if df.empty:
        return pd.DataFrame(columns=[
            'teamId', 'period', 'timeInPeriod', 'shotType', 'xCoord', 'yCoord',
            'shooterId', 'shooterName', 'goalieId', 'goalieName', 'typeDescKey',
            'zoneCode'
        ])
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
    return df[['teamId', 'period', 'timeInPeriod', 'shotType', 'xCoord', 'yCoord',
               'shooterId', 'shooterName', 'goalieId', 'goalieName', 'typeDescKey',
               'zoneCode']]


def get_dataframe_from_data(season):
    dataScrap = LNHDataScrapper()
    data_csv = f"{dataScrap.dest_folder}/{season}.csv"

    if os.path.exists(data_csv):
        return pd.read_csv(data_csv)

    games = dataScrap.open_data(season)
    result = pd.DataFrame([])

    for game in games:
        players = pd.json_normalize(game.get("rosterSpots", []))
        plays = game.get("plays", [])

        # Extraire 3 types d'événements de tirs
        df_sog   = get_dataframe_from_shot_on_goal_event(players, plays)
        df_goal  = get_dataframe_from_goal_event(players, plays)
        df_miss  = get_dataframe_from_missed_shot_event(players, plays)
        df_other = get_dataframe_from_other_event(players,plays)
        # Concat locale
        dfs = [df_sog, df_goal, df_miss, df_other]
# Garder seulement les DataFrames non vides
        dfs = [df for df in dfs if not df.empty]

        df_game = pd.concat(dfs, ignore_index=True)
        # df_game = pd.concat([df_sog, df_goal, df_miss,df_other], ignore_index=True) old version
        if df_game.empty:
            continue

        # Enrichir avec idGame & season
        game_id = game.get("id") or game.get("gamePk") or game.get("gameId")
        game_season = game.get("season")
        df_game["idGame"] = game_id
        df_game["season"] = game_season

        # Mapper l'abréviation d'équipe
        meta = _game_team_meta(game)
        df_game["teamAbbr"] = df_game["teamId"].apply(lambda tid: _owner_to_abbr(tid, meta))

        # Nettoyage minimal
        df_game = df_game.dropna(subset=["xCoord", "yCoord"])
        df_game = df_game.sort_values(by=['period', 'timeInPeriod'])
        result = pd.concat([result, df_game], ignore_index=True)

    # Sauvegarde complète avec colonnes utiles aux cartes
    if not result.empty:
        result.to_csv(data_csv, index=False)
    return result

def get_seasons_dataframe(begin,end):
    df = pd.DataFrame()
    for season in [f"{y}{y+1}" for y in range(begin, end)]:
        df = pd.concat([df,get_dataframe_from_data(season)], ignore_index=True)
    return df

if __name__ == "__main__":
    for season in [f"{y}{y+1}" for y in range(2016, 2024)]:
        df = get_dataframe_from_data(season)
        print(f"Season {season} -> {len(df)} lignes sauvegardées.")
    print("All seasons processed.")
