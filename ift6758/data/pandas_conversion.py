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


def get_playerName_from_game(game_players_data : pd.DataFrame,searchedPlayerId):
    '''Permet de récupérer le nom d'un joueur en donnant l'ID'''
    # game_players_data = game_players_data.set_index('playerId')
    searchedPlayer = game_players_data.loc[game_players_data['playerId']==searchedPlayerId]
    if len(searchedPlayer)==0:
        return None
    first_name = searchedPlayer['firstName.default'].iloc[0]
    last_name = searchedPlayer['lastName.default'].iloc[0]
    return f"{first_name} {last_name}"



def get_dataframe_from_shot_on_goal_event(players,plays):
    df = pd.DataFrame(plays)
    filterValue = ['shot-on-goal'] #'blocked-shot','missed-shot', 'goal'
    details = pd.json_normalize(df["details"])
    period = pd.json_normalize(df['periodDescriptor'])
    df = df[df["typeDescKey"].isin(filterValue) ]
    df["xCoord"]=details["xCoord"]
    df["yCoord"]=details["yCoord"]
    df["shooterId"] = details["shootingPlayerId"]
    df["shooterName"] = df["shooterId"].apply(lambda pid: get_playerName_from_game(players, pid))
    df["goalieId"] = details["goalieInNetId"]
    df["goalieName"] = df["goalieId"].apply(lambda pid: get_playerName_from_game(players, pid))
    df["shotType"] = details["shotType"]
    df["period"] = period["number"]
    df["teamId"] = details["eventOwnerTeamId"]
    return df[['teamId','period','timeInPeriod','shotType','xCoord','yCoord','shooterId','shooterName','goalieId','goalieName','typeDescKey']]

def get_dataframe_from_goal_event(players,plays):
    df = pd.DataFrame(plays)
    filterValue = ['goal'] #'blocked-shot','missed-shot', 'goal'
    details = pd.json_normalize(df["details"])
    period = pd.json_normalize(df['periodDescriptor'])
    df = df[df["typeDescKey"].isin(filterValue) ]
    df["xCoord"]=details["xCoord"]
    df["yCoord"]=details["yCoord"]
    df["shooterId"] = details["scoringPlayerId"]
    df["shooterName"] = df["shooterId"].apply(lambda pid: get_playerName_from_game(players, pid))
    df["goalieId"] = details["goalieInNetId"]
    df["goalieName"] = df["goalieId"].apply(lambda pid: get_playerName_from_game(players, pid))
    df["shotType"] = details["shotType"]
    df["period"] = period["number"]
    df["teamId"] = details["eventOwnerTeamId"]
    return df[['teamId','period','timeInPeriod','shotType','xCoord','yCoord','shooterId','shooterName','goalieId','goalieName','typeDescKey']]



def get_dataframe_from_data(season):
        dataScrap = LNHDataScrapper()
        data_csv = f"{dataScrap.dest_folder}/{season}.csv"
        if os.path.exists(data_csv):
            result = pd.read_csv(data_csv)
        else:
            data = dataScrap.open_data(season)
            result = pd.DataFrame([])
            for game in data:
                players = pd.json_normalize(game["rosterSpots"])
                shotongoalEvent = get_dataframe_from_shot_on_goal_event(players,game["plays"])
                goalEvent = get_dataframe_from_goal_event(players,game["plays"])
                result = pd.concat([result,shotongoalEvent,goalEvent])
            result.to_csv(f"{dataScrap.dest_folder}/{season}.csv")
        return result



if __name__ == "__main__":
    for season in [f"{y}{y+1}" for y in range(2016, 2024)]:
        get_dataframe_from_data(season)
        print(f"Season {season} data processed and saved as CSV.")
    print("All seasons processed.")