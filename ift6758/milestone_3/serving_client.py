import json
import requests
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class ServingClient:
    """
    Client for communicating with the Flask prediction backend.

    Methods preserved:
      - predict(X)
      - logs()
      - download_registry_model()
    """

    def __init__(self, ip: str = "127.0.0.1", port: int = 5000, features=None):
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"[ServingClient] Base URL = {self.base_url}")

        # The GameClient may pass a feature list, but the Flask API decides mapping.
        self.features = features if features is not None else ["distance_net"]


    # -----------------------------------------------------------
    # 1. /predict
    # -----------------------------------------------------------
    def predict(self, X: pd.DataFrame, model_key: str = None) -> pd.DataFrame:
        """
        Sends each row of X to the Flask /predict endpoint (one sample at a time).
        Flask returns: {"prediction": float}

        Args:
            X : pd.DataFrame of features
            model_key : optional model name ("logreg-distance", "logreg-distance-angle")

        Returns:
            pd.DataFrame with column: 'model_output'
        """

        url = f"{self.base_url}/predict"
        results = []

        logger.info(
            f"[ServingClient] POST {url} with {len(X)} samples (sent one-by-one)"
        )

        # Loop through samples because Flask expects one sample per request
        for idx, row in X.iterrows():
            payload = row.to_dict()

            # Add model key if provided
            if model_key is not None:
                payload["model"] = model_key

            try:
                response = requests.post(url, json=payload)
            except requests.exceptions.RequestException as e:
                logger.error(f"[ServingClient] Request failed: {e}")
                raise

            if response.status_code != 200:
                logger.error(f"/predict returned {response.status_code}: {response.text}")
                raise RuntimeError(response.text)

            out = response.json()

            if "prediction" not in out:
                raise ValueError(f"Malformed API response: {out}")

            # Append the numeric probability
            results.append(out["prediction"])

        df_out = X.copy()
        df_out["model_output"] = results

        return df_out


    # -----------------------------------------------------------
    # 2. /logs  
    # -----------------------------------------------------------
    def logs(self) -> dict:
        """
        GET /logs
        Returns:
            dict or string with plaintext log contents
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

        # Flask returns raw log text (not JSON)
        try:
            return response.json()
        except:
            return {"logs": response.text}


    # -----------------------------------------------------------
    # 3. /download_registry_model  
    # -----------------------------------------------------------
    def download_registry_model(self, workspace: str, model: str, version: str) -> dict:
        """
        POST /download_registry_model
        Args:
            workspace: wandb workspace name
            model: model_key ("logreg-distance", ...)
            version: string ("v0", "v1", ...)

        Returns:
            dict of Flask response
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
            logger.error(
                f"[ServingClient] /download_registry_model returned {response.status_code}"
            )
            raise RuntimeError(response.text)

        return response.json()