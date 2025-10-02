import requests
import json
import os
import pandas as pd

class LNHDataScrapper:
    def __init__(self):
        self.dest_folder = "./ressources/"
        if os.path.exists(self.dest_folder) ==False:
            os.mkdir(self.dest_folder)

    def get_one_game(self,game_id,save:bool =False):
        response = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play")
        try:
            if response.status_code == 200:
                data = response.json()
                if save == True:
                    filename = f"{self.dest_folder}{game_id}.json"
                    with open(filename,'w') as json_file:
                        json.dump(data,json_file,indent=4)
                return data
                

            else:
                print(f"Error: API request failed with status code {response.status_code}")
                print(response.text)  # Print the error message if available
                raise NameError(f"Error: API request failed with status code {response.status_code}.\n {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"An error occurred during the API request: {e}")

    def get_season_data(self,season):
        data : list = []
        filename = f"{self.dest_folder}{season}.json"
        max_game = 20
        try:
            for i in range(1,max_game):
                data.append(self.get_one_game(f"{season[0:4]}02{i:04}"))
                if os.name == 'nt':  # For Windows
                    os.system('cls')
                else:  # For macOS and Linux
                    os.system('clear')
                print(f"Download season {season}: {round((i/max_game)*100, 2)} %")
            
            with open(filename,'w') as json_file:
                if os.name == 'nt':  # For Windows
                    os.system('cls')
                else:  # For macOS and Linux
                    os.system('clear')
                print(f"Download finished ! \n Saving data (don't close program)")
                json.dump(data,json_file,indent=4)
                json_file.close()
                
            return data
        except Exception as e:
            print(f"An unexpected error occurred during data downloading: {e}")
    
    def open_data(self,season):
        filename = f"{self.dest_folder}{season}.json"
        if os.path.exists(filename):
            try:
                with open(filename,'r') as f:
                    data = json.load(f)
                   
            except FileNotFoundError:
                print(f"Error: The file '{filename}' was not found.")
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from '{filename}'. Check if the file contains valid JSON.")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
        else :
            data = self.get_season_data(season)
        return data

    def get_dataframe_from_data(self,season):
        data = self.open_data(season)
        df = pd.DataFrame(data[0]["plays"])
        
        players = pd.DataFrame(data[0]["rosterSpots"])
        players["firstName"] = pd.json_normalize(players["firstName"])["default"]
        players["lastName"] =pd.json_normalize(players["firstName"])["default"]
        players = players.set_index("playerId")
        
        filterValue = ['shot-on-goal','goal'] #'blocked-shot','missed-shot', 
        details = pd.json_normalize(df["details"])
        period = pd.json_normalize(df['periodDescriptor'])
        df = df[df["typeDescKey"].isin(filterValue) ]
        df["xCoord"]=details["xCoord"]
        df["yCoord"]=details["yCoord"]
        df["shooterId"] = details["shootingPlayerId"]
        # df["game_id"]= data[0]["id"]
        # print(df.head(10))
        return df


if __name__ == "__main__":
    lnhdata = LNHDataScrapper()
    # lnhdata.get_season_data("2017")
    # lnhdata.open_data("2017")
    df = lnhdata.get_dataframe_from_data("20242025")
    print(df.head())
    