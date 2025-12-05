import os
from pathlib import Path
import flask
import logging
import threading # NOUVEAU: Pour la gestion sécurisée de l'accès aux variables globales (MODEL_CACHE_LOCK)
from flask import Flask, jsonify, request, abort
import sklearn
import pandas as pd
import joblib
import pickle
import numpy as np
import sys 

import ift6758
# wandb importé de manière lazy pour éviter les conflits pydantic au démarrage

# ===============================================================================
# CONFIGURATION GLOBALE
# ===============================================================================

# Informations de base sur les modèles supportés
MODELS_INFO  =  {
    "logreg-distance": {
        "model": "logreg-distance",              
        "filename": "logreg_distance.pkl",     
        "workspace":"qi-li-1-universit-de-montr-al",
        "version": "v0" # Version par défaut si non spécifiée
    },
    "logreg-distance-angle": {
        "model": "logreg-distance-angle",
        "filename": "logreg_distance_angle.pkl",
        "workspace":"qi-li-1-universit-de-montr-al",
        "version": "v0"
    }
}
VALID_MODELS = list(MODELS_INFO.keys()) 
PROJECT = "IFT6758-2025"
LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")

# --- VARIABLES GLOBALES POUR GÉRER PLUSIEURS MODÈLES CONCURRENTS ---
""" Cache des modèles
MODEL_CACHE = {
    "logreg-distance": {
        "object": <LogisticRegression(max_iter=100)>,  : objet modèle réel chargé
        "version": "v0",
        "filename": "logreg_distance.pkl",
        "workspace": "qi-li-1-universit-de-montr-al",
        "model": "logreg-distance"
    },
    "logreg-distance-angle": {
        "object": <LogisticRegression(max_iter=100)>,
        "version": "v0",
        "filename": "logreg_distance_angle.pkl",
        "workspace": "qi-li-1-universit-de-montr-al",
        "model": "logreg-distance-angle"
    }
}
"""
MODEL_CACHE = {} 
# Verrou pour sécuriser l'écriture dans MODEL_CACHE lors de la mise à jour (Hot-Swap)
MODEL_CACHE_LOCK = threading.Lock() 
# Clé de modèle utilisée par défaut si non spécifiée dans la requête /predict
DEFAULT_MODEL_KEY = "logreg-distance" 


# Dossier pour stocker les modèles téléchargés
# Utiliser un chemin relatif sécurisé à côté du fichier app.py
MODELS_DIR = Path(__file__).parent.parent.parent / "ift6758" / "models_saved" 

# Initialize Flask app
app = Flask(__name__)

# ===============================================================================
# INITIALISATION ET JOURNALISATION
# ===============================================================================

# Création des dossiers nécessaires
log_dir = os.path.dirname(LOG_FILE) if os.path.dirname(LOG_FILE) else '.'
if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)
if not MODELS_DIR.exists():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
# Configuration de la journalisation (logging)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
app.logger.info("Application starting up...")

# Initialisation Wandb (lazy import pour éviter les conflits pydantic)
wandb_api_key = os.environ.get("WANDB_API_KEY")
if wandb_api_key:
    app.logger.info("WANDB_API_KEY found. Wandb will be initialized on first use.")
else:
    app.logger.warning("WANDB_API_KEY not found. Wandb registry operations may fail.")       

# ===============================================================================
# FONCTIONS UTILITAIRES DE MODÈLE
# ===============================================================================

def download_model_from_wandb(model_key, version):
    """
    Télécharge un modèle depuis Wandb Model Registry.
    Returns:
        tuple: (model_dir, model_file_path, model_info)
    """
    try:
        # Import lazy de wandb pour éviter les problèmes au démarrage
        import wandb
        
        # Initialiser wandb si ce n'est pas déjà fait
        wandb_api_key = os.environ.get("WANDB_API_KEY")
        if wandb_api_key:
            try:
                wandb.login(key=wandb_api_key, relogin=True, verify=False)
                app.logger.info("Wandb initialized successfully.")
            except Exception as e:
                app.logger.error(f"Failed to login to Wandb: {e}")
                raise
                      
        if model_key not in VALID_MODELS:
           raise ValueError(f"Modèle '{model_key}' non supporté.")
    
        temp_model_info = MODELS_INFO[model_key].copy()
        temp_model_info["version"] = version 
        
        workspace = temp_model_info["workspace"]
        filename = temp_model_info["filename"]
                                             
        artifact_name = f"{workspace}/{PROJECT}/{model_key}:{version}"
        
        # 1. Utiliser l'API Wandb pour télécharger
        api = wandb.Api()
        artifact = api.artifact(artifact_name, type="model")
        
        # 2. Créer le répertoire de destination : models_saved/model_key/version/
        target_dir = MODELS_DIR / model_key / version
        target_dir.mkdir(parents=True, exist_ok=True)
        app.logger.info(f"Downloading model to: {target_dir}")
        
        # 3. Télécharger dans le répertoire organisé
        model_dir = artifact.download(root=str(target_dir))
        # 3. Vérifier le fichier
        downloaded_path = Path(target_dir) / filename
        
        if not downloaded_path.exists():
            # Fallback si le nom de fichier par défaut n'est pas trouvé
            all_pkl_files = list(Path(target_dir).glob("*.pkl"))
            if all_pkl_files:
                downloaded_path = all_pkl_files[0]
                app.logger.warning(f"Using fallback file: {downloaded_path.name}")
            else:
                raise FileNotFoundError(f"No .pkl file found in downloaded artifact for {artifact_name}")
        
        
        return model_dir, downloaded_path, temp_model_info
        
    except Exception as e:
        app.logger.error(f"Error downloading model {model_key}:{version} from Wandb: {e}", exc_info=False)
        raise
  
def load_model_local(file_path):
     """
     Charge le modèle sérialisé (joblib) depuis le chemin spécifié.
     
     Returns:
          model
     """
     if not Path(file_path).exists():
        raise FileNotFoundError(f"Model file not found: {file_path}")

     # Utilisation de joblib.load, recommandé pour scikit-learn
     model = joblib.load(file_path)

     return model
            
# ===============================================================================
# POINTS DE TERMINAISON DE L'API
# ===============================================================================

@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""
    
    if not os.path.exists(LOG_FILE):
        message = f"Le fichier de log '{LOG_FILE}' n'existe pas encore.\n"
        message += "Le fichier sera créé lorsque le serveur sera démarré.\n"
        
        response = flask.make_response(message)
        response.headers["Content-Type"] = "text/plain"
        return response, 200 
    
    try:
        with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            log_content = f.read()
        
        if not log_content.strip():
            log_content = "Le fichier de log existe mais est vide.\nAucun log n'a encore été généré."
            
        response = flask.make_response(log_content)
        response.headers["Content-Type"] = "text/plain; charset=utf-8"
        return response    
    except Exception as e:
        return jsonify({'error': f"Erreur lors de la lecture du fichier journal: {e}"}), 500
    

@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Télécharge et charge un modèle depuis Wandb Model Registry.
    Le modèle est stocké dans le MODEL_CACHE, permettant le Hot-Swapping
    et le support de plusieurs modèles simultanés.
    """
    global MODEL_CACHE, MODEL_CACHE_LOCK
    
    # 1. Récupérer et valider les données JSON
    try:
        json_data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "JSON invalide ou absent"}), 400
    
    model_key = json_data.get("model")
    version = json_data.get("version", "v0") 
    
    if not model_key:
        return jsonify({"error": "Le nom du modèle (key 'model') est requis."}), 400
    
    if model_key not in VALID_MODELS:
        app.logger.warning(f"Requested model '{model_key}' not in VALID_MODELS list.")
        return jsonify({"error": f"Modèle '{model_key}' non supporté"}), 400
    
    requested_version_key = f"{model_key}:{version}"

    # 2. Vérifier si la version demandée est déjà en cache mémoire
    if model_key in MODEL_CACHE and MODEL_CACHE[model_key]["version"] == version:
        app.logger.info(f"Model {requested_version_key} already in cache. No action needed.")
        return jsonify({
            "status": "already_loaded",
            "message": "Model already loaded and active in cache.",
            "model_info": {
                "model": MODEL_CACHE[model_key]["model"],
                "version": MODEL_CACHE[model_key]["version"],
                "filename": MODEL_CACHE[model_key]["filename"],
                "workspace": MODEL_CACHE[model_key]["workspace"]                
            }
        })

    # 3. Vérifier si le modèle existe en disque local (disque) dans la nouvelle structure
    model_info = MODELS_INFO[model_key]
    filename = model_info["filename"]
    
    # Chemin attendu : models_saved/model_key/version/filename
    local_model_path = MODELS_DIR / model_key / version / filename
    
    if local_model_path.exists():
        app.logger.info(f"Model file found in local disque: {local_model_path}")
        try:
            new_model_object = load_model_local(local_model_path)
            
            # Mettre à jour le cache en mémoire
            with MODEL_CACHE_LOCK:
                MODEL_CACHE[model_key] = {
                    "object": new_model_object,
                    "version": version,
                    "filename": filename,
                    "workspace": model_info["workspace"],
                    "model": model_key
                }
            
            app.logger.info(f"Model {requested_version_key} loaded from local cache")
            return jsonify({
                "status": "success",
                "message": "Model loaded from local disque",
                "model_info": {
                    "model": model_key,
                    "version": version,
                    "filename": filename,
                    "workspace": model_info["workspace"]
                },
                "source": "local_disque"
            })
        except Exception as e:
            app.logger.warning(f"Failed to load from local disque: {e}, will download from Wandb") 
    
    # 4. Sinon Télécharger le modèle depuis Wandb
    try:
        # model_info contient les infos mises à jour (y compris la version demandée)
        model_dir, model_file_path, new_model_info = download_model_from_wandb(model_key, version=version)
        app.logger.info(f"Model {requested_version_key} downloaded successfully to {model_dir}")        
    except Exception as e:
        error_msg = f"Failed to download model {requested_version_key}: {str(e)}"
        app.logger.warning(error_msg)
        
        # Logique de résilience : On ne brise pas l'application. On garde le cache existant.
        return jsonify({"status": "download_failed", "error": error_msg}), 500

    # 5. Tenter de Charger le nouveau modèle (Mise à jour du cache sécurisée)
    try:
        # Charger le modèle dans une variable locale temporaire
        new_model_object = load_model_local(model_file_path)
        
        # Utiliser un verrou pour garantir que la mise à jour du dictionnaire est atomique
        with MODEL_CACHE_LOCK:
            MODEL_CACHE[model_key] = {
                "object": new_model_object, # L'objet modèle chargé
                "version": new_model_info["version"],
                "filename": new_model_info["filename"],
                "workspace": new_model_info["workspace"],
                "model": new_model_info["model"]
            }
       
        app.logger.info(f"SUCCESS: Model {requested_version_key} updated in cache.")
        
    except Exception as e:
        error_msg = f"Failed to load downloaded model {requested_version_key}: {str(e)}"
        app.logger.error(error_msg)       
        # Logique de résilience : Le cache n'est pas mis à jour avec la version corrompue.
        return jsonify({"status": "load_failed", "error": error_msg}), 500
    
    # 6. Retourner la réponse de succès
    response = {
        "status": "success",
        "message": "Model downloaded and loaded into cache successfully",
        "model_info": {
            "model": MODEL_CACHE[model_key]["model"],
            "version": MODEL_CACHE[model_key]["version"],
            "filename": MODEL_CACHE[model_key]["filename"],
            "workspace": MODEL_CACHE[model_key]["workspace"]            
        }
    }
    
    app.logger.info(response)
    return jsonify(response)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Utilise le modèle spécifié dans le cache pour la prédiction.
    La requête JSON doit inclure la clé 'model'.
    """
    
    # Récupérer les données
    json_data = request.get_json()
    if not json_data:
        return jsonify({"error": "No JSON data provided"}), 400

    # Tenter d'obtenir la clé du modèle depuis le JSON, sinon utiliser la clé par défaut
    model_key = json_data.get("model", DEFAULT_MODEL_KEY)
    
    # 1. Vérifier si le modèle demandé est dans le cache
    if model_key not in MODEL_CACHE:
        app.logger.warning(f"Prediction requested for unloaded model: {model_key}")
        return jsonify({"error": f"Model '{model_key}' is not loaded in cache. Use /download_registry_model first."}), 400
    
    # 2. Récupérer le modèle du cache (accès en lecture, donc sûr)
    current_model = MODEL_CACHE[model_key]["object"]

    # 3. Préparer les données pour la prédiction (exclure "model", garder les features)
    predict_data = {k: v for k, v in json_data.items() if k != "model"}
    
    # 4. Mapper et filtrer les features selon le modèle utilisé
    mapped_data = {}
    
    # Définir les features requises par modèle
    model_features = {
        "logreg-distance": ["distance_net"],
        "logreg-distance-angle": ["distance_net", "angle_net"]
    }
    
    required_features = model_features.get(model_key, [])
    
    if required_features:
        # Mapping des noms alternatifs
        feature_aliases = {
            "distance_net": ["distance", "distance_net"],
            "angle_net": ["angle", "angle_net"]
        }
        
        for feature in required_features:
            found = False
            for alias in feature_aliases.get(feature, [feature]):
                if alias in predict_data:
                    mapped_data[feature] = predict_data[alias]
                    found = True
                    break
            
            if not found:
                app.logger.error(f"Missing feature '{feature}' for model '{model_key}'")
                return jsonify({
                    "error": f"Model '{model_key}' requires '{feature}' feature (or aliases: {feature_aliases.get(feature, [feature])})"
                }), 400
    else:
        # Pour les autres modèles, utiliser les données telles quelles
        mapped_data = predict_data

    # 5. Prédiction
    try:
        X_new = pd.DataFrame([mapped_data])
        
        # Faire la prédiction
        prediction = current_model.predict_proba(X_new)[:, 1]
        
        return jsonify({
            "prediction": float(prediction[0]), 
            "model_used": MODEL_CACHE[model_key]["model"],
            "version_used": MODEL_CACHE[model_key]["version"]
        })
    
    except KeyError as e:
        app.logger.error(f"Missing required feature for model {model_key}: {e}")
        return jsonify({"error": f"Missing required feature: {e}"}), 400
    except ValueError as e:
        app.logger.error(f"Invalid data format for model {model_key}: {e}")
        return jsonify({"error": f"Invalid data format: {e}"}), 400
    except Exception as e:
        app.logger.error(f"Prediction error for model {model_key}: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ===============================================================================
# DÉMARRAGE DU SERVEUR
# ===============================================================================
if __name__ == "__main__":
    # Configuration du logging pour voir les erreurs
    print("="*60)
    print("Starting Flask server...")
    print("URL: http://0.0.0.0:5000")
    print("="*60)
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    
    # Démarrer le serveur sans reloader pour éviter les problèmes
    # Note: Utiliser 0.0.0.0 au lieu de 127.0.0.1 pour éviter les problèmes de binding
    try:
        from werkzeug.serving import run_simple
        run_simple('0.0.0.0', 5000, app, use_reloader=False, use_debugger=False, threaded=True)
    except Exception as e:
        print(f"ERROR: Failed to start server: {e}")
        import traceback
        traceback.print_exc()
        raise
