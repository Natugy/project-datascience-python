"""
Application Streamlit pour le Milestone 3 - IFT 6758
Interface interactive pour visualiser les prédictions de buts attendus (xG) en temps réel
"""

import streamlit as st
import pandas as pd
import requests
import sys
import os
import json
from pathlib import Path

# Ajouter le chemin du module ift6758 au sys.path
project_root = Path(__file__).parent.parent  # Remonter à milestone_3
sys.path.insert(0, str(project_root))

from serving_client import ServingClient
from game_client import GameClient

BACKEND = os.getenv("BACKEND_URL", "127.0.0.1")

# ============================================================================
# CONFIGURATION DE LA PAGE
# ============================================================================
st.set_page_config(
    page_title="NHL xG Tracker",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# INITIALISATION DE L'ÉTAT DE LA SESSION
# ============================================================================
# Configuration des valeurs par défaut pour l'état de la session
DEFAULT_SESSION_STATE = {
    "serving_client": None,
    "game_client": None,
    "current_model": None,
    "current_game_id": None,
    "game_data": None,
    "predictions_df": pd.DataFrame(),
    "home_xg": 0.0,
    "away_xg": 0.0,
    "auto_connected": False
}


# Initialiser toutes les valeurs de session si elles n'existent pas
for key, default_value in DEFAULT_SESSION_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# ============================================================================
# CONNEXION AUTOMATIQUE AU SERVICE
# ============================================================================
if not st.session_state.auto_connected:
    try:
        st.session_state.serving_client = ServingClient(ip=BACKEND, port=5000)
        st.session_state.game_client = GameClient(serving_client=st.session_state.serving_client)
        st.session_state.auto_connected = True
    except Exception as e:
        st.error(f"Erreur de connexion automatique au service Flask: {e}")
        st.info("Assurez-vous que le service Flask est démarré sur 127.0.0.1:5000")

# ============================================================================
# STYLE CSS PERSONNALISÉ
# ============================================================================
st.markdown("""
<style>
    /* Amélioration des métriques */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 600;
    }
    
    /* Amélioration des sous-titres */
    .stSubheader {
        color: #1f77b4;
        font-weight: 600;
        margin-top: 1.5rem;
    }
    
    /* Amélioration des DataFrames */
    [data-testid="stDataFrame"] {
        border-radius: 8px;
    }
    
    /* Boutons plus modernes */
    .stButton button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    /* Boutons primaires en gris foncé */
    .stButton button[kind="primary"] {
        background-color: #424242;
        color: white;
    }
    
    .stButton button[kind="primary"]:hover {
        background-color: #616161;
    }
    
    /* Espacement des colonnes */
    [data-testid="column"] {
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# TITRE ET EN-TÊTE
# ============================================================================
st.markdown("<h1 style='text-align: center; color: #1f77b4; margin-bottom: 0;'>Hockey Visualization App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; margin-top: 0;'>Prédictions de buts attendus (xG) en temps réel</p>", unsafe_allow_html=True)
st.markdown("---")

# ============================================================================
# SIDEBAR - CONFIGURATION DU MODÈLE
# ============================================================================
with st.sidebar:
    # Section pour le chargement du modèle
    st.markdown("### Configuration du Modèle")
    
    workspace = st.text_input(
        "Workspace Wandb",
        value="qi-li-1-universit-de-montr-al",
        help="Nom du workspace Wandb contenant vos modèles"
    )
    
    model_version = st.text_input(
        "Version du modèle",
        value="v0",
        help="Version du modèle à charger (ex: v0, v1, latest)"
    )
    
    model_name = st.selectbox(
        "Modèle de prédiction",
        options=["logreg-distance", "logreg-distance-angle"],
        help="Sélectionnez le modèle à utiliser",
        index=0
    )
    
    if st.button("Charger le modèle", use_container_width=True, type="primary", key="load_model_btn"):
        if st.session_state.serving_client is None:
            st.error("Veuillez d'abord vous connecter au service!")
        else:
            try:
                with st.spinner("Téléchargement du modèle en cours..."):
                    response = st.session_state.serving_client.download_registry_model(
                        workspace=workspace,
                        model=model_name,
                        version=model_version
                    )
                    st.session_state.current_model = model_name
                    st.success(f"Modèle chargé avec succès!")
                    with st.expander("Détails de la réponse"):
                        st.json(response)
            except Exception as e:
                st.error(f"Erreur: {e}")
    
    # Affichage du modèle actuel avec style amélioré
    if st.session_state.current_model:
        st.markdown(f"""
        <div style='background: #2e7d32; 
                    padding: 1rem; border-radius: 8px; margin: 1rem 0;'>
            <p style='color: white; margin: 0; font-size: 0.85rem; opacity: 0.9;'>Modèle actif</p>
            <p style='color: white; margin: 0; font-size: 1.1rem; font-weight: 600;'>{st.session_state.current_model}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Outils avancés dans un expander
    with st.expander("Outils avancés"):
        if st.button("Voir les logs", use_container_width=True):
            if st.session_state.serving_client is None:
                st.error("Service non connecté")
            else:
                try:
                    logs = st.session_state.serving_client.logs()
                    st.text_area("Logs", value=logs.get("logs", str(logs)), height=200)
                except Exception as e:
                    st.error(f"Erreur: {e}")
        
        if st.button("Recharger les IDs", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache vidé!")
            st.rerun()

# ============================================================================
# FONCTION POUR CHARGER LES IDS DE MATCHS
# ============================================================================
@st.cache_data
def load_game_ids():
    """
    Charge tous les IDs de matchs disponibles depuis les fichiers JSON.
    
    Returns:
        list: Liste triée des IDs de matchs (plus récents en premier)
    """
    # Obtenir le chemin absolu du répertoire de données
    # De streamlit_app.py -> ift6758 -> milestone_3 -> streamlit -> project root -> data -> raw
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent.parent
    data_dir = project_root / "data" / "raw"
    
    # Vérifier que le répertoire existe
    if not data_dir.exists():
        st.warning(f"Répertoire de données non trouvé: {data_dir}")
        return []
    
    game_ids = set()  # Utiliser un set directement pour éviter les doublons
    
    # Lire tous les fichiers JSON
    for json_file in data_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Support pour différents formats de données
                if isinstance(data, list):
                    # Format liste de matchs
                    for game in data:
                        if isinstance(game, dict) and 'id' in game:
                            game_ids.add(str(game['id']))
                elif isinstance(data, dict) and 'id' in data:
                    # Format match unique
                    game_ids.add(str(data['id']))
                    
        except json.JSONDecodeError as e:
            st.warning(f"Format JSON invalide dans {json_file.name}: {e}")
        except Exception as e:
            st.warning(f"Erreur lors de la lecture de {json_file.name}: {e}")
    
    # Trier les IDs par ordre décroissant (plus récents en premier)
    return sorted(game_ids, reverse=True)

# ============================================================================
# FONCTIONS UTILITAIRES POUR LE CALCUL DES XG
# ============================================================================
def calculate_xg_by_team(predictions_df, game_client):
    """
    Calcule les xG (Expected Goals) par équipe.
    
    Args:
        predictions_df: DataFrame contenant les prédictions avec model_output et teamId
        game_client: Instance de GameClient avec les données du match
        
    Returns:
        tuple: (home_xg, away_xg) - xG pour l'équipe domicile et visiteur
    """
    if predictions_df.empty:
        return 0.0, 0.0
    
    # Récupérer les IDs d'équipe depuis game_client
    if game_client and game_client.cached_game_data:
        game_data = game_client.cached_game_data
        home_id = game_data["homeTeam"]["id"]
        away_id = game_data["awayTeam"]["id"]
        
        # Calculer les xG par équipe si teamId est disponible
        if "teamId" in predictions_df.columns:
            home_xg = predictions_df[
                predictions_df["teamId"] == home_id
            ]["model_output"].sum()
            
            away_xg = predictions_df[
                predictions_df["teamId"] == away_id
            ]["model_output"].sum()
            
            return home_xg, away_xg
    
    # Fallback: diviser approximativement
    total_xg = predictions_df["model_output"].sum()
    return total_xg / 2, total_xg / 2

# ============================================================================
# MAIN - INTERFACE DE JEU
# ============================================================================
st.markdown("### Suivi du Match en Direct")

# Charger les IDs de matchs disponibles
try:
    available_game_ids = load_game_ids()
    if not available_game_ids:
        st.warning("Aucun fichier JSON trouvé dans data/raw/. Utilisation de l'ID par défaut.")
        available_game_ids = ["2021020329"]  # Valeur par défaut si aucun fichier trouvé
    else:
        # Afficher le nombre de matchs trouvés
        with st.expander("Information sur les matchs disponibles", expanded=False):
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.metric("Matchs disponibles", len(available_game_ids))
            with col_info2:
                st.caption(f"**Plage d'IDs:** {available_game_ids[-1]} → {available_game_ids[0]}")
except Exception as e:
    st.error(f"Erreur lors du chargement des IDs de matchs: {e}")
    st.warning("Utilisation de l'ID par défaut: 2021020329")
    available_game_ids = ["2021020329"]  # Valeur par défaut

# Sélection de l'ID du jeu avec interface simplifiée
col1, col2 = st.columns([3, 1])

with col1:
    # Mode de sélection
    input_mode = st.radio(
        "Mode de sélection",
        options=["Liste des matchs", "Saisie manuelle"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    if input_mode == "Liste des matchs":
        game_id = st.selectbox(
            "Sélectionner un match",
            options=available_game_ids,
            index=0,
            help="393 matchs disponibles (2016-2023)"
        )
    else:
        game_id = st.text_input(
            "ID du match NHL",
            placeholder="Ex: 2024020001",
            help="Format: SSSSTTGGGG (Saison-Type-Numéro)"
        )
        if not game_id:
            game_id = available_game_ids[0]  # Valeur par défaut

with col2:
    st.write("")  # Espacement
    st.write("")  # Espacement
    ping_button = st.button("Charger le Match", use_container_width=True, type="primary", key="load_match_btn")

# Détecter automatiquement le changement d'ID et réinitialiser
if st.session_state.current_game_id is not None and st.session_state.current_game_id != game_id:
    # Réinitialiser automatiquement
    st.session_state.current_game_id = None
    st.session_state.game_data = None
    st.session_state.predictions_df = pd.DataFrame()
    st.session_state.home_xg = 0.0
    st.session_state.away_xg = 0.0
    if st.session_state.game_client:
        st.session_state.game_client.last_seen_event_id = -1
    st.info(f"Nouveau match: {game_id} - Cliquez sur 'Charger le Match'")

# Traitement du ping game
if ping_button:
    if st.session_state.game_client is None:
        st.error("Service Flask non connecté!")
    elif st.session_state.current_model is None:
        st.error("Aucun modèle chargé! Utilisez la sidebar pour charger un modèle.")
    else:
        try:
            with st.spinner("Chargement des données du match..."):
                # Ping le jeu pour obtenir les nouvelles données
                metadata, df_pred = st.session_state.game_client.ping_game(game_id)
                
                # Sauvegarder les métadonnées
                st.session_state.game_data = metadata
                st.session_state.current_game_id = game_id
                
                # Si de nouveaux événements ont été trouvés
                if df_pred is not None and not df_pred.empty:
                    # Ajouter les nouvelles prédictions au DataFrame existant
                    st.session_state.predictions_df = pd.concat(
                        [st.session_state.predictions_df, df_pred],
                        ignore_index=True
                    )
                    
                    st.success(f"{len(df_pred)} nouveaux événement(s) ajouté(s)!")
                else:
                    st.info("Aucun nouvel événement (match déjà chargé)")
                
        except Exception as e:
            st.error(f"Erreur lors du ping: {e}")
            import traceback
            st.code(traceback.format_exc())

# ============================================================================
# AFFICHAGE DES DONNÉES DU MATCH
# ============================================================================
if st.session_state.game_data:
    metadata = st.session_state.game_data
    
    # En-tête du match avec style amélioré
    st.markdown("---")
    st.markdown("### Informations du Match")
    
    # Créer trois colonnes pour l'affichage
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown(f"<div style='text-align: center; padding: 1rem; background: #1f77b4; border-radius: 10px; color: white;'>"
                   f"<h3 style='margin: 0; color: white;'>{metadata.get('home_team', 'N/A')}</h3>"
                   f"<h1 style='margin: 0.5rem 0; color: white;'>{metadata.get('home_score', 0)}</h1>"
                   f"<p style='margin: 0; opacity: 0.9;'>Domicile</p></div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"<div style='text-align: center; padding: 1rem;'>"
                   f"<p style='font-size: 1.2rem; margin: 0; color: #666;'>Période</p>"
                   f"<h2 style='margin: 0.5rem 0; color: #1f77b4;'>{metadata.get('period', 'N/A')}</h2>"
                   f"<p style='font-size: 1rem; color: #888;'>{metadata.get('time_remaining', 'N/A')}</p></div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"<div style='text-align: center; padding: 1rem; background: #d62728; border-radius: 10px; color: white;'>"
                   f"<h3 style='margin: 0; color: white;'>{metadata.get('away_team', 'N/A')}</h3>"
                   f"<h1 style='margin: 0.5rem 0; color: white;'>{metadata.get('away_score', 0)}</h1>"
                   f"<p style='margin: 0; opacity: 0.9;'>Visiteur</p></div>", unsafe_allow_html=True)
    
    # Calcul des xG (buts attendus) totaux par équipe
    if not st.session_state.predictions_df.empty:
        home_xg, away_xg = calculate_xg_by_team(
            st.session_state.predictions_df, 
            st.session_state.game_client
        )
        total_xg = home_xg + away_xg
        
        # Affichage des xG avec design amélioré
        st.markdown("---")
        st.markdown("### Expected Goals (xG)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            home_score = metadata.get('home_score', 0)
            delta_home = home_score - home_xg
            arrow = '↑' if delta_home > 0 else '↓'
            st.markdown(f"<div style='background: #e3f2fd; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #1f77b4;'>"
                       f"<p style='font-size: 0.9rem; color: #666; margin: 0;'>{metadata.get('home_team', 'Domicile')}</p>"
                       f"<h2 style='color: #1f77b4; margin: 0.5rem 0;'>{home_xg:.2f} xG</h2>"
                       f"<p style='color: {'#2e7d32' if delta_home > 0 else '#c62828'}; font-size: 0.9rem; margin: 0;'>{arrow} {abs(delta_home):.2f}</p></div>", unsafe_allow_html=True)
        
        with col2:
            away_score = metadata.get('away_score', 0)
            delta_away = away_score - away_xg
            arrow = '↑' if delta_away > 0 else '↓'
            st.markdown(f"<div style='background: #ffebee; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #d62728;'>"
                       f"<p style='font-size: 0.9rem; color: #666; margin: 0;'>{metadata.get('away_team', 'Visiteur')}</p>"
                       f"<h2 style='color: #d62728; margin: 0.5rem 0;'>{away_xg:.2f} xG</h2>"
                       f"<p style='color: {'#2e7d32' if delta_away > 0 else '#c62828'}; font-size: 0.9rem; margin: 0;'>{arrow} {abs(delta_away):.2f}</p></div>", unsafe_allow_html=True)

# ============================================================================
# AFFICHAGE DU DATAFRAME DES PRÉDICTIONS
# ============================================================================
if not st.session_state.predictions_df.empty:
    st.markdown("---")
    st.markdown("### Détails des Événements de Tir")
    
    # Options d'affichage
    show_all = st.checkbox("Afficher toutes les colonnes", value=False)
    
    if show_all:
        st.dataframe(
            st.session_state.predictions_df,
            use_container_width=True,
            height=350
        )
    else:
        # Afficher seulement les colonnes pertinentes avec renommage
        display_cols = ["distance_net", "angle_net", "empty_net", "model_output"]
        available_cols = [col for col in display_cols if col in st.session_state.predictions_df.columns]
        
        df_display = st.session_state.predictions_df[available_cols].copy()
        df_display.columns = ["Distance (ft)", "Angle (°)", "Filet vide", "Probabilité xG"]
        
        st.dataframe(
            df_display,
            use_container_width=True,
            height=350
        )
    
    # Statistiques rapides avec design amélioré
    st.markdown("---")
    st.markdown("### Statistiques des Tirs")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Nombre total de tirs", len(st.session_state.predictions_df))
    
    with col2:
        avg_distance = st.session_state.predictions_df["distance_net"].mean()
        st.metric("Distance moyenne", f"{avg_distance:.2f} ft")
    
    with col3:
        if "angle_net" in st.session_state.predictions_df.columns:
            avg_angle = st.session_state.predictions_df["angle_net"].mean()
            st.metric("Angle moyen", f"{avg_angle:.2f}°")
    
    with col4:
        if "empty_net" in st.session_state.predictions_df.columns:
            empty_net_shots = st.session_state.predictions_df["empty_net"].sum()
            st.metric("Tirs filet vide", int(empty_net_shots))

else:
    # Afficher un message seulement si aucun match n'a jamais été chargé
    if st.session_state.current_game_id is None:
        st.info("Sélectionnez un ID de match ci-dessus et cliquez sur 'Charger le Match' pour commencer!")

# ============================================================================
# BOUTON DE RÉINITIALISATION
# ============================================================================
st.markdown("---")
if st.button("Réinitialiser les données du match"):
    st.session_state.predictions_df = pd.DataFrame()
    st.session_state.game_data = None
    st.session_state.current_game_id = None
    st.session_state.home_xg = 0.0
    st.session_state.away_xg = 0.0
    st.success("Données réinitialisées!")
    st.rerun()

# ============================================================================
# PIED DE PAGE
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>IFT 6758 - Milestone 3</strong></p>
    <p>Application de prédiction de buts attendus (Expected Goals) pour la NHL</p>
</div>
""", unsafe_allow_html=True)
