import requests
import json
import os
import pandas as pd
import threading
from tqdm import tqdm

class DownloadException(Exception):
    """Exception custom pour les matchs qui n'existe pas"""
    def __init__(self, message="Match inexistant"):
        self.message = message
        super().__init__(self.message)

class LNHDataScrapper:
    def __init__(self):
        self.dest_folder = "./ressources/"
        if os.path.exists(self.dest_folder) ==False:
            os.mkdir(self.dest_folder)

    def generate_playoff_game_ids(self,season):
        game_ids = []
        game_type = "03"  # playoffs

        
        rounds = {
            1: 8, 
            2: 4,  
            3: 2,  
            4: 1  
        }

        for rnd, matchups in rounds.items():
            for matchup in range(1, matchups + 1):
                for game in range(1, 8):  # jusqu’à 7 matchs possibles
                    game_number = f"0{rnd}{matchup}{game}"
                    game_id = f"{season[0:4]}{game_type}{game_number}"
                    game_ids.append(game_id)
        
        return game_ids


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
                
            elif response.status_code ==404:
                raise DownloadException()
            else:
                raise NameError(f"Error: API request failed with status code {response.status_code}.\n {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"An error occurred during the API request: {e}")

    def get_season_data(self,season):
        data : list = []
        filename = f"{self.dest_folder}{season}.json"
        max_game = 1353
        for i in tqdm(range(1,max_game+1),desc= "Téléchargement de la saison régulière"):
            try:
                game = self.get_one_game(f"{season[0:4]}02{i:04}")
                data.append(game)
            except DownloadException as e:
                break
        playoff_ids = self.generate_playoff_game_ids(season[0:4])
        for game_id in tqdm(playoff_ids, desc="Téléchargement des playoffs "):
            try:
                game = self.get_one_game(game_id)
                data.append(game)
            except DownloadException as e:
                continue
        
        with open(filename,'w') as json_file:
            print(f"Sauvegarde des données de la saison {season} en cours, veuillez ne pas interrompre le programme")
            json.dump(data,json_file,indent=4)
            json_file.close()
            
        return data
        
    
    def open_data(self,season):
        filename = f"{self.dest_folder}{season}.json"
        data : list = []
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

    


if __name__ == "__main__":
    scrap = LNHDataScrapper()
    data = scrap.open_data("20182019")

