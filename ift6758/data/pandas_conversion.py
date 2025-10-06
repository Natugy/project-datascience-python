import pandas as pd
from .data_scrapping import LNHDataScrapper


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
    return df[['teamId','period','shotType','xCoord','yCoord','shooterId','shooterName','goalieId','goalieName','typeDescKey']]

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
    return df[['teamId','period','shotType','xCoord','yCoord','shooterId','shooterName','goalieId','goalieName','typeDescKey']]



def get_dataframe_from_data(season):
        dataScrap = LNHDataScrapper()
        data = dataScrap.open_data(season)
        result = pd.DataFrame([])
        for game in data:
            players = pd.json_normalize(game["rosterSpots"])
            shotongoalEvent = get_dataframe_from_shot_on_goal_event(players,game["plays"])
            goalEvent = get_dataframe_from_goal_event(players,game["plays"])
            result = pd.concat([result,shotongoalEvent,goalEvent])
               
        return result



if __name__ == "__main__":
    df = get_dataframe_from_data("20242025")
    print(df)
    