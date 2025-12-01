import json
import requests
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class ServingClient:
    """
    A lightweight Python client that wraps HTTP requests to the Flask
    model-serving backend.

    Exposes:
        - predict(X)
        - logs()
        - download_registry_model(...)
    """

    def __init__(self, ip: str = "127.0.0.1", port: int = 5000, features=None):
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"[ServingClient] Base URL = {self.base_url}")

        if features is None:
            features = ["distance_net"]
        self.features = features


    # -----------------------------------------------------------
    # 1. /predict
    # -----------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Sends a POST request to /predict with the DataFrame payload.
        Returns prediction results as a DataFrame aligned with X.

        Args:
            X : pd.DataFrame with input features

        Returns:
            pd.DataFrame with a "prediction" column appended
        """

        url = f"{self.base_url}/predict"
        logger.info(f"[ServingClient] POST {url} with {len(X)} rows")

        # Convert DF to JSON-compatible dict (NOT string)
        payload = json.loads(X.to_json())

        try:
            response = requests.post(url, json=payload)
        except requests.exceptions.RequestException as e:
            logger.error(f"[ServingClient] Request failed: {e}")
            raise

        if response.status_code != 200:
            logger.error(f"[ServingClient] /predict returned {response.status_code}")
            raise RuntimeError(response.text)

        result = response.json()

        # The backend typically returns: {"predictions": [p1, p2, ...]}
        preds = result.get("predictions")
        if preds is None:
            raise ValueError("Malformed response: 'predictions' missing")

        df_out = X.copy()
        df_out["model_output"] = preds

        return df_out


    # -----------------------------------------------------------
    # 2. /logs
    # -----------------------------------------------------------
    def logs(self) -> dict:
        """
        GET /logs
        Returns logs from the Flask server.

        Returns:
            dict with a list of log messages
        """

        url = f"{self.base_url}/logs"
        logger.info(f"[ServingClient] GET {url}")

        try:
            response = requests.get(url)
        except requests.exceptions.RequestException as e:
            logger.error(f"[ServingClient] Request failed: {e}")
            raise

        if response.status_code != 200:
            logger.error(f"[ServingClient] /logs returned {response.status_code}")
            raise RuntimeError(response.text)

        return response.json()


    # -----------------------------------------------------------
    # 3. /download_registry_model
    # -----------------------------------------------------------
    def download_registry_model(self, workspace: str, model: str, version: str) -> dict:
        """
        POST /download_registry_model
        Used to swap in a new model version at runtime.

        Args:
            workspace: wandb workspace name
            model: model name in wandb registry
            version: version string

        Returns:
            dict indicating success/failure of the swap
        """

        url = f"{self.base_url}/download_registry_model"
        logger.info(f"[ServingClient] POST {url} with {workspace}/{model}:{version}")

        payload = {
            "workspace": workspace,
            "model": model,
            "version": version,
        }

        try:
            response = requests.post(url, json=payload)
        except requests.exceptions.RequestException as e:
            logger.error(f"[ServingClient] Request failed: {e}")
            raise

        if response.status_code != 200:
            logger.error(f"[ServingClient] /download_registry_model returned {response.status_code}")
            raise RuntimeError(response.text)

        return response.json()