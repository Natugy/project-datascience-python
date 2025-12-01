import requests
import numpy as np
import pandas as pd
import logging

from pandas_conversion import (
    get_dataframe_from_shot_on_goal_event,
    get_dataframe_from_goal_event
    )
# You may adjust imports depending on your repo structure
from features_simple import (
    clean_dataframe,
    detect_empty_net_from_situation
    )
from serving_client import ServingClient


logger = logging.getLogger(__name__)


class GameClient:
    """
    Client to fetch NHL live game data, track new events, extract features,
    and send them to the prediction service.

    Implements:
        - ping_game(game_id)
        - get_new_shots_for_prediction()
        - get_game_metadata()
    """

    NHL_URL = "https://api-web.nhle.com/v1/gamecenter/{}/play-by-play"

    def __init__(self, serving_client: ServingClient):
        # A ServingClient instance used to call /predict
        self.serving_client = serving_client

        # Internal state
        self.last_seen_event_id = -1
        self.cached_game_data = None
        self.cached_game_id = None

    # ----------------------------------------------------------
    # Fetch raw play-by-play from the NHL API
    # ----------------------------------------------------------
    def fetch_game(self, game_id: str) -> dict:
        url = self.NHL_URL.format(game_id)
        logger.info(f"[GameClient] Fetching game {game_id} from NHL API")

        response = requests.get(url)
        if response.status_code != 200:
            raise RuntimeError(f"NHL API returned {response.status_code}: {response.text}")

        game_json = response.json()

        # Reset tracking state if changing game
        if game_id != self.cached_game_id:
            self.cached_game_id = game_id
            self.last_seen_event_id = -1

        self.cached_game_data = game_json
        return game_json

    # ----------------------------------------------------------
    # Extract useful game info for the Streamlit UI
    # ----------------------------------------------------------
    def get_game_metadata(self) -> dict:
        game = self.cached_game_data
        if game is None:
            return {}

        home = game["homeTeam"]["commonName"]["default"]
        away = game["awayTeam"]["commonName"]["default"]
        period = game["periodDescriptor"]["number"]
        time_remaining = game["clock"]["timeRemaining"]

        home_score = game["homeTeam"]["score"]
        away_score = game["awayTeam"]["score"]

        return {
            "home_team": home,
            "away_team": away,
            "home_score": home_score,
            "away_score": away_score,
            "period": period,
            "time_remaining": time_remaining,
        }

    # ----------------------------------------------------------
    # Extract only NEW shot events
    # ----------------------------------------------------------
    def get_new_shots_for_prediction(self) -> pd.DataFrame:
        """
        Returns a DataFrame containing only NEW events since last ping.
        """
        plays = self.cached_game_data.get("plays", [])
        if not plays:
            return pd.DataFrame()

        # Filter only new events
        new_events = [p for p in plays if p["eventId"] > self.last_seen_event_id]

        if not new_events:
            return pd.DataFrame()

        # Identify maximum new event ID to update tracker later
        new_last_id = max(p["eventId"] for p in new_events)

        # Filter only shot events
        new_shots = [p for p in new_events if p["typeDescKey"] in ["shot-on-goal", "goal"]]

        if not new_shots:
            # Still update last seen
            self.last_seen_event_id = new_last_id
            return pd.DataFrame()

        # Convert events to features
        df = self._convert_events_to_features(new_shots)

        # Update tracker
        self.last_seen_event_id = new_last_id

        return df

    # ----------------------------------------------------------
    # Convert raw shot events into model features
    # ----------------------------------------------------------
    def _convert_events_to_features(self, events: list) -> pd.DataFrame:
        """
        Takes new shot events and converts them into the SAME feature format
        used during training (distance, angle, empty_net).
        """

        # We need the full game context to compute distance/angle
        full_game = self.cached_game_data
        home_id = full_game["homeTeam"]["id"]
        away_id = full_game["awayTeam"]["id"]

        players = pd.json_normalize(full_game.get("rosterSpots", []))
        plays = events

        # Extract SOG and goals from your pandas_conversion utilities
        df_sog = get_dataframe_from_shot_on_goal_event(players, plays, home_id, away_id)
        df_goal = get_dataframe_from_goal_event(players, plays, home_id, away_id)
        df = pd.concat([df_sog, df_goal], ignore_index=True)

        if df.empty:
            return df

        # Compute empty-net label
        df = detect_empty_net_from_situation(df)

        # Compute distance + angle
        goal_x = np.where(df["xCoord"] > 0, 89, -89)
        goal_y = 0

        df["distance_net"] = np.sqrt(
            (goal_x - df["xCoord"]) ** 2 + 
            (goal_y - df["yCoord"]) ** 2)
        
        df["angle_net"] = abs(
            np.degrees(np.arctan2(df["yCoord"], (goal_x - df["xCoord"])))
        )

        df["is_goal"] = (df["typeDescKey"] == "goal").astype(int)

        # Keep same columns as training features
        return df[["distance_net", "angle_net", "empty_net"]]

    # ----------------------------------------------------------
    # Main entrypoint used by Streamlit
    # ----------------------------------------------------------
    def ping_game(self, game_id: str):
        """
        Fetches the game, finds new shots, extracts features,
        and sends them to the prediction service.

        Returns:
            metadata: dict
            df_pred: DataFrame OR None if no new events
        """

        self.fetch_game(game_id)

        metadata = self.get_game_metadata()
        new_df = self.get_new_shots_for_prediction()

        if new_df.empty:
            return metadata, None

        # Call prediction service
        df_pred = self.serving_client.predict(new_df)

        return metadata, df_pred