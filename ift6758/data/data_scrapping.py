"""
Script de téléchargement des données NHL via l'API NHL.
"""

import requests
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict, Optional

class DownloadException(Exception):
    """Exception pour les matchs inexistants ou erreurs de téléchargement."""
    pass


class LNHDataScrapper:
    """
    Classe pour télécharger les données NHL depuis l'API officielle.
    Utilise le threading pour accélérer les téléchargements.
    """
    
    def __init__(self, dest_folder: Optional[str] = None):
        """
        Initialise le scrapper avec le dossier de destination.
        
        Args:
            dest_folder: Dossier de sauvegarde (par défaut: data/raw/ depuis la racine du projet)
        """
        if dest_folder is None:
            # Trouver la racine du projet
            project_root = Path(__file__).parent.parent.parent
            self.dest_folder = project_root / "data" / "raw"
        else:
            self.dest_folder = Path(dest_folder)
        
        self.dest_folder.mkdir(parents=True, exist_ok=True)
        self.api_base_url = "https://api-web.nhle.com/v1/gamecenter"
        
        # Cache pour éviter les requêtes répétées
        self._cache = {}

    def generate_playoff_game_ids(self, season: str) -> List[str]:
        """
        Génère tous les IDs de matchs de playoffs possibles pour une saison.
        
        Args:
            season: Saison au format "20162017"
            
        Returns:
            Liste des game_ids de playoffs
        """
        game_ids = []
        game_type = "03"  # Playoffs
        
        # Structure des playoffs: 4 rondes avec nombre décroissant de séries
        rounds = {
            1: 8,  # 8 séries en 1ère ronde
            2: 4,  # 4 séries en 2ème ronde (demi-finales de conférence)
            3: 2,  # 2 séries en 3ème ronde (finales de conférence)
            4: 1   # 1 série en finale de la Coupe Stanley
        }
        
        for rnd, matchups in rounds.items():
            for matchup in range(1, matchups + 1):
                for game in range(1, 8):  # Jusqu'à 7 matchs possibles par série
                    game_number = f"0{rnd}{matchup}{game}"
                    game_id = f"{season[:4]}{game_type}{game_number}"
                    game_ids.append(game_id)
        
        return game_ids


    def get_one_game(self, game_id: str, save: bool = False) -> Optional[Dict]:
        """
        Télécharge les données d'un match spécifique.
        
        Args:
            game_id: ID du match (ex: "2016020001")
            save: Si True, sauvegarde le JSON individuellement
            
        Returns:
            Données du match en dict, ou None si erreur
            
        Raises:
            DownloadException: Si le match n'existe pas (404)
        """
        # Vérifier le cache
        if game_id in self._cache:
            return self._cache[game_id]
        
        url = f"{self.api_base_url}/{game_id}/play-by-play"
        
        try:
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                self._cache[game_id] = data

                # Treat empty JSON `{}` as invalid
                if not data:
                    return None
                
                if save:
                    filename = self.dest_folder / f"{game_id}.json"
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=4)
                
                return data
            
            elif response.status_code == 404:
                raise DownloadException(f"Match {game_id} n'existe pas")
            
            else:
                return None
                
        except requests.exceptions.Timeout:
            return None
        except requests.exceptions.RequestException as e:
            return None

    def _download_game_batch(self, game_ids: List[str]) -> List[Dict]:
        """
        Télécharge plusieurs matchs en parallèle.
        
        Args:
            game_ids: Liste des IDs de matchs à télécharger
            
        Returns:
            Liste des données de matchs (non-None uniquement)
        """
        games = []
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Soumettre toutes les tâches
            future_to_id = {
                executor.submit(self.get_one_game, gid): gid 
                for gid in game_ids
            }
            
            # Récupérer les résultats avec barre de progression
            with tqdm(total=len(game_ids), desc="Téléchargement") as pbar:
                for future in as_completed(future_to_id):
                    game_id = future_to_id[future]
                    try:
                        data = future.result()
                        if data is not None:
                            games.append(data)
                    except DownloadException:
                        pass  # Match inexistant, normal
                    except Exception as e:
                        pass  # Ignorer les erreurs inattendues
                    finally:
                        pbar.update(1)
        
        return games
    
    def get_season_data(self, season: str, force_redownload: bool = False) -> List[Dict]:
        """
        Télécharge toutes les données d'une saison (régulière + playoffs).
        
        Args:
            season: Saison au format "20162017"
            force_redownload: Si True, re-télécharge même si le fichier existe
            
        Returns:
            Liste de tous les matchs de la saison
        """
        filename = self.dest_folder / f"{season}.json"
        
        # Vérifier si le fichier existe et n'est pas vide
        if filename.exists() and not force_redownload:
            file_size = filename.stat().st_size
            if file_size > 1024:  # Plus de 1KB = fichier valide
                print(f"Fichier {season}.json existe ({file_size/1024:.1f} KB), chargement...")
                return self.open_data(season)
            else:
                print(f"Fichier {season}.json vide ou corrompu, re-téléchargement...")
                filename.unlink()  # Supprimer le fichier vide

        print(f"Téléchargement de la saison {season}...")

        # Générer les IDs de la saison régulière
        max_game = 1353
        
        year1 = season[:4]  # Ex: "2022"
        year2 = season[4:]  # Ex: "2023"
        
        # Générer les IDs avec l'année de début (format classique)
        regular_ids = [f"{year1}02{i:04}" for i in range(1, max_game + 1)]
        
        # Pour les saisons >= 2022, ajouter aussi les IDs avec l'année de fin
        if int(year1) >= 2022:
            regular_ids_year2 = [f"{year2}02{i:04}" for i in range(1, max_game + 1)]
            regular_ids.extend(regular_ids_year2)
            print(f"Saison récente: test des deux formats ({year1}... et {year2}...)")
        
        # Générer les IDs des playoffs
        playoff_ids = self.generate_playoff_game_ids(season)
        
        # Télécharger en parallèle
        regular_games = self._download_game_batch(regular_ids)

        playoff_games = self._download_game_batch(playoff_ids)
        
        # Combiner les données
        all_games = regular_games + playoff_games
        
        # Sauvegarder
        print(f"Sauvegarde de {len(all_games)} matchs dans {filename.name}...")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(all_games, f, indent=4)
        
        file_size = filename.stat().st_size / 1024
        print(f"Saison {season} téléchargée: {len(all_games)} matchs ({file_size:.1f} KB)")
        
        return all_games
        
    def open_data(self, season: str) -> List[Dict]:
        """
        Charge les données d'une saison depuis le fichier JSON.
        Télécharge automatiquement si le fichier n'existe pas.
        
        Args:
            season: Saison au format "20162017"
            
        Returns:
            Liste des matchs de la saison
        """
        filename = self.dest_folder / f"{season}.json"
        
        if filename.exists():
            try:
                print(f"Chargement de {filename}...")
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data
            except json.JSONDecodeError:
                print(f"Erreur de décodage JSON dans {filename}")
                return []
            except Exception as e:
                print(f"Erreur lors du chargement: {e}")
                return []
        else:
            print(f"Fichier {filename} introuvable, téléchargement...")
            return self.get_season_data(season)


def main():
    """Fonction principale pour télécharger toutes les saisons."""
    scrapper = LNHDataScrapper()
    
    # Saisons à télécharger (2016-17 à 2023-24)
    seasons = [f"{y}{y+1}" for y in range(2016, 2024)]
    
    print("=" * 70)
    print(f"  TÉLÉCHARGEMENT DE {len(seasons)} SAISONS NHL (2016-2024)")
    print("=" * 70)
    print()
    
    total_games = 0
    successful_seasons = 0
    empty_seasons = []

    for i, season in enumerate(seasons, 1):
        print(f"\n[{i}/{len(seasons)}] Saison {season[:4]}-{season[4:]}")
        print("-" * 70)
        try:
            data = scrapper.open_data(season)
            if len(data) > 0:
                total_games += len(data)
                successful_seasons += 1
                print(f"{season}: {len(data)} matchs chargés")
            else:
                empty_seasons.append(season)
                print(f"{season}: Aucun match (fichier vide)")
        except Exception as e:
            empty_seasons.append(season)
            print(f"{season}: Erreur - {e}")

    print()
    print("=" * 70)
    print("  RÉSUMÉ DU TÉLÉCHARGEMENT")
    print("=" * 70)
    print(f"Saisons réussies: {successful_seasons}/{len(seasons)}")
    print(f"Total matchs téléchargés: {total_games:,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
