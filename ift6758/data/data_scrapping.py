"""
Script de téléchargement des données NHL via l'API NHL.
Optimisé pour performance avec requêtes parallèles et caching intelligent.
"""

import requests
import json
import os
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict, Optional

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
            dest_folder: Dossier de sauvegarde (par défaut: data/raw/)
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
        
        logger.info(f"Scrapper initialisé. Destination: {self.dest_folder}")

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
                
                if save:
                    filename = self.dest_folder / f"{game_id}.json"
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=4)
                
                return data
            
            elif response.status_code == 404:
                raise DownloadException(f"Match {game_id} n'existe pas")
            
            else:
                logger.error(f"Erreur API {response.status_code} pour {game_id}")
                return None
                
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout pour {game_id}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur requête pour {game_id}: {e}")
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
                        logger.error(f"Erreur inattendue pour {game_id}: {e}")
                    finally:
                        pbar.update(1)
        
        return games
    
    def get_season_data(self, season: str) -> List[Dict]:
        """
        Télécharge toutes les données d'une saison (régulière + playoffs).
        
        Args:
            season: Saison au format "20162017"
            
        Returns:
            Liste de tous les matchs de la saison
        """
        filename = self.dest_folder / f"{season}.json"
        
        # Si le fichier existe déjà, le charger
        if filename.exists():
            logger.info(f"Fichier {season}.json existe déjà, chargement...")
            return self.open_data(season)
        
        logger.info(f"Téléchargement de la saison {season}...")
        
        # Générer les IDs de la saison régulière
        max_game = 1353
        regular_ids = [f"{season[:4]}02{i:04}" for i in range(1, max_game + 1)]
        
        # Générer les IDs des playoffs
        playoff_ids = self.generate_playoff_game_ids(season)
        
        # Télécharger en parallèle
        logger.info(f"Saison régulière (jusqu'à {max_game} matchs)")
        regular_games = self._download_game_batch(regular_ids)

        logger.info(f"Playoffs (jusqu'à {len(playoff_ids)} matchs possibles)")
        playoff_games = self._download_game_batch(playoff_ids)
        
        # Combiner les données
        all_games = regular_games + playoff_games
        
        # Sauvegarder
        logger.info(f"Sauvegarde de {len(all_games)} matchs dans {filename}...")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(all_games, f, indent=4)
        
        logger.info(f"Saison {season} téléchargée: {len(all_games)} matchs")
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
                logger.info(f"Chargement de {filename}...")
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"{len(data)} matchs chargés pour {season}")
                return data
            except json.JSONDecodeError:
                logger.error(f"Erreur de décodage JSON dans {filename}")
                return []
            except Exception as e:
                logger.error(f"Erreur lors du chargement: {e}")
                return []
        else:
            logger.info(f"Fichier {filename} introuvable, téléchargement...")
            return self.get_season_data(season)


def main():
    """Fonction principale pour télécharger toutes les saisons."""
    scrapper = LNHDataScrapper()
    
    # Saisons à télécharger (2016-17 à 2023-24)
    seasons = [f"{y}{y+1}" for y in range(2016, 2024)]
    
    logger.info(f"=== Téléchargement de {len(seasons)} saisons NHL ===")
    
    for season in seasons:
        try:
            data = scrapper.open_data(season)
            logger.info(f"{season}: {len(data)} matchs")
        except Exception as e:
            logger.error(f"Erreur pour {season}: {e}")
    
    logger.info("=== Téléchargement terminé ===")


if __name__ == "__main__":
    main()
