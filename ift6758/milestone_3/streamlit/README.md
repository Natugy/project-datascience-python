# Hockey Visualization App - Milestone 3 Partie 5

## Description

Application Streamlit "Hockey Visualization App" pour visualiser en temps réel les prédictions de buts attendus (Expected Goals - xG) pour les matchs de la NHL. Elle s'intègre avec le service Flask de prédiction et affiche les données de manière interactive.

Cette application répond à toutes les exigences du Milestone 3 - Partie 5 du cours IFT 6758.

## Structure des Fichiers

```
ift6758/milestone_3/streamlit/
├── streamlit_app.py                # Application principale
├── check_setup.py                  # Vérification de l'environnement
├── test_streamlit_components.py    # Tests automatisés
├── start_streamlit.ps1             # Script de lancement Streamlit
├── start_flask.ps1                 # Script de lancement Flask
└── .streamlit_config/
    └── config.toml                 # Configuration Streamlit (thème, serveur)

ift6758/milestone_3/
├── serving_client.py               # Client Flask API
├── game_client.py                  # Client NHL API (avec teamId pour xG par équipe)
├── pandas_conversion.py
└── features_simple.py

ift6758/backend/
└── app.py                          # Service Flask
```

## Installation

### Prérequis
- Python 3.8+
- Environnement virtuel `ift6758-venv` activé
- Clé API Wandb configurée
- Packages requis dans `requirements.txt`

```powershell
cd D:\Bureau\project-datascience-python
pip install -r requirements.txt
```

Packages principaux : `streamlit`, `pandas`, `requests`, `waitress`, `wandb`, `scikit-learn`

## Démarrage Rapide

### Option 1 : Scripts PowerShell

#### Terminal 1 : Démarrer Flask
```powershell
cd D:\Bureau\project-datascience-python\ift6758\milestone_3\streamlit
.\start_flask.ps1
```

Actions automatiques :
- Activation de l'environnement virtuel
- Vérification/saisie de WANDB_API_KEY
- Démarrage du service Flask sur http://0.0.0.0:5000

#### Terminal 2 : Démarrer Streamlit
```powershell
cd D:\Bureau\project-datascience-python\ift6758\milestone_3\streamlit
.\start_streamlit.ps1
```

Actions automatiques :
- Activation de l'environnement virtuel
- Vérification/saisie de WANDB_API_KEY
- Configuration de STREAMLIT_CONFIG_DIR
- Démarrage de Streamlit sur http://localhost:8501

### Option 2 : Commandes Manuelles

#### Terminal 1 : Flask
```powershell
& D:\Bureau\project-datascience-python\ift6758-venv\Scripts\Activate.ps1
$env:WANDB_API_KEY="votre_cle_api"
cd D:\Bureau\project-datascience-python\ift6758\backend
waitress-serve --listen=0.0.0.0:5000 app:app
```

#### Terminal 2 : Streamlit
```powershell
& D:\Bureau\project-datascience-python\ift6758-venv\Scripts\Activate.ps1
$env:STREAMLIT_CONFIG_DIR=".streamlit_config"
cd D:\Bureau\project-datascience-python\ift6758\milestone_3\streamlit
streamlit run streamlit_app.py
```

### Option 3 : Docker Compose

```powershell
cd D:\Bureau\project-datascience-python
$env:WANDB_API_KEY="votre_cle_api"
docker-compose up --build
```

Services disponibles :
- Streamlit : http://localhost:8501
- Flask : http://localhost:5000

**Note pour Docker** : Utiliser `serving` comme IP au lieu de `127.0.0.1`

## Utilisation de l'Application

### Workflow Complet

#### 1. Configuration du Service
1. Ouvrir http://localhost:8501
2. Dans la sidebar "Configuration du Service de Prédiction" :
   - IP : Sélectionner `127.0.0.1` (local, par défaut) ou `serving` (Docker)
   - Port : `5000`
   - Cliquer "Se connecter au service"
3. Confirmation : Message de connexion réussie

#### 2. Chargement du Modèle
Dans la sidebar "Chargement du Modèle" :
- Workspace : `qi-li-1-universit-de-montr-al`
- Modèle : `logreg-distance` ou `logreg-distance-angle`
- Version : `v0`
- Cliquer "Télécharger le modèle"

Le service Flask télécharge le modèle depuis Wandb et l'échange à chaud.

#### 3. Suivi d'un Match
Dans la zone principale :
- Sélectionner un ID de match dans la liste déroulante (chargés depuis `data/raw/*.json`)
- Cliquer "Ping Game"

L'application affiche :
- Métadonnées du match (équipes, score, période, temps)
- Expected Goals (xG) par équipe avec différence vs score réel
- DataFrame détaillé des événements de tir
- Statistiques des tirs

#### 4. Pings Suivants
- Re-cliquer "Ping Game" pour récupérer les nouveaux événements
- Le système filtre automatiquement les doublons via `last_seen_event_id`
- Pour un match terminé : tous les événements sont récupérés au 1er ping

## Fonctionnalités Implémentées

### Interface Utilisateur

#### Sidebar
- **Configuration du Service** : Connexion au service Flask (sélection IP: 127.0.0.1 ou serving)
- **Chargement du Modèle** : Téléchargement depuis Wandb Model Registry
- **Affichage des Logs** : Consultation des logs du service Flask

#### Zone Principale
- **Suivi du Match** : Sélection d'ID depuis une liste déroulante et ping
- **Informations du Match** : Équipes, score, période, temps restant
- **Expected Goals (xG)** :
  - xG équipe domicile (avec différence vs score réel)
  - xG équipe visiteur (avec différence vs score réel)
  - xG total du match
- **DataFrame des Événements** : Toutes les features + probabilités
- **Statistiques** : Total tirs, distance/angle moyens, filet vide
- **Réinitialisation** : Bouton pour réinitialiser les données

### Fonctionnalités Techniques

#### Calcul des xG par Équipe
- Utilise `teamId` retourné par `game_client.py`
- Sépare les tirs de l'équipe domicile et visiteur
- Compare avec le score réel (delta affiché)

#### Filtrage des Événements
- `GameClient` maintient `last_seen_event_id`
- Évite les doublons lors de pings multiples
- Accumule les prédictions dans `st.session_state`

#### Gestion de l'État
Utilise `st.session_state` pour :
- Connexion au service (ServingClient, GameClient)
- Modèle actuellement chargé
- ID du match en cours
- Prédictions accumulées (`predictions_df`)
- Métadonnées du match (`game_data`)
- Tracking xG par équipe

## Architecture

```
┌──────────────────────────────────────────┐
│        Streamlit Frontend                │
│  ┌────────────┐    ┌─────────────────┐  │
│  │  Sidebar   │    │   Main Area     │  │
│  │  - Config  │    │   - Game Info   │  │
│  │  - Models  │    │   - xG Metrics  │  │
│  │  - Logs    │    │   - DataFrame   │  │
│  └────────────┘    └─────────────────┘  │
└──────────────────────────────────────────┘
         │                      │
         ↓                      ↓
  ┌─────────────┐      ┌──────────────┐
  │ServingClient│      │ GameClient   │
  │ - predict() │      │ - ping_game()│
  │ - logs()    │      │ - get_shots()│
  │ - download()│      └──────────────┘
  └─────────────┘             │
         │                    ↓
         ↓            ┌──────────────┐
  ┌─────────────┐    │   NHL API    │
  │  Flask API  │    │ api-web.nhle │
  └─────────────┘    └──────────────┘
         │
         ↓
  ┌─────────────┐
  │Wandb Registry│
  └─────────────┘
```

## IDs de Match pour Tests

Format : `SSSSTTGGGG`
- `SSSS` : Saison (ex: 2021 pour 2021-2022)
- `TT` : Type (02 = saison régulière, 03 = playoffs)
- `GGGG` : Numéro du match

### Exemples
- `2021020329` - Match saison régulière 2021-2022
- `2021020330` - Match saison régulière 2021-2022
- `2022030411` - Match playoffs 2022
- `2023020001` - Premier match saison 2023-2024

## Tests et Validation

### Test des composants
```powershell
python test_streamlit_components.py
```

Tests automatisés :
1. **ServingClient** : Connexion, logs, téléchargement de modèle
2. **GameClient** : Récupération de données, métadonnées, événements
3. **Workflow Complet** : Configuration, chargement, ping, calcul xG

Résultat attendu :
```
[PASS] - ServingClient
[PASS] - GameClient
[PASS] - Workflow Complet
[SUCCESS] Tous les tests ont réussi!
```

## Résolution de Problèmes

### Erreur de connexion au service
**Symptôme** : "Cannot connect to service" ou erreur de connexion, ou `NameResolutionError` pour l'hôte 'serving'

**Solutions** :
1. Vérifier que Flask est démarré : http://127.0.0.1:5000/logs
2. **IMPORTANT** : Vérifier l'adresse IP dans la sidebar
   - **Local** : Utiliser `127.0.0.1` (PAS `serving`)
   - **Docker** : Utiliser `serving` (PAS `127.0.0.1`)
3. Vérifier le port : `5000`
4. Vérifier les logs Flask pour erreurs

### Module not found
**Symptôme** : `ModuleNotFoundError: No module named 'streamlit'`

**Solutions** :
```powershell
cd D:\Bureau\project-datascience-python
pip install -r requirements.txt
```

### WANDB_API_KEY not found
**Symptôme** : Erreur lors du téléchargement de modèle

**Solutions** :
```powershell
# PowerShell
$env:WANDB_API_KEY="votre_cle_api"

# Ou créer un fichier .env
echo "WANDB_API_KEY=votre_cle_api" > .env
```

### Erreur de téléchargement du modèle
**Symptôme** : "Model not found in registry"

**Solutions** :
- Vérifier la clé API Wandb
- Vérifier workspace, nom du modèle et version
- S'assurer d'avoir les permissions d'accès
- Consulter les logs du service Flask

### Aucun nouvel événement détecté
**Symptôme** : "Aucun nouvel événement de tir détecté"

**Explication** : Comportement normal pour un match terminé. Tous les événements sont récupérés au 1er ping. Les pings suivants ne retourneront rien car le système filtre les doublons.

**Solution** : Utiliser un autre ID de match ou cliquer "Réinitialiser les données"

### Erreurs d'importation
**Symptôme** : `ImportError` pour serving_client ou game_client

**Solutions** :
- Vérifier que vous êtes dans le bon dossier
- Vérifier que les fichiers existent dans `../` (milestone_3/)
- Vérifier le `sys.path` dans streamlit_app.py

### Service Flask ne démarre pas
**Symptôme** : Erreur lors de `waitress-serve`

**Solutions** :
```powershell
# Réinstaller waitress
pip install --upgrade waitress

# Ou utiliser Flask directement (dev uniquement)
python app.py
```

## Configuration

### Fichier .streamlit_config/config.toml

```toml
[theme]
primaryColor = "#C8102E"        # Rouge NHL
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[server]
port = 8501
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
```

**Note** : Le dossier `.streamlit_config` est utilisé au lieu de `.streamlit`. La variable `STREAMLIT_CONFIG_DIR` est définie automatiquement par `start_streamlit.ps1`.

### Chemins Relatifs

L'application utilise des chemins relatifs pour l'importation :

```python
# Dans streamlit_app.py
project_root = Path(__file__).parent.parent  # Remonte à milestone_3
sys.path.insert(0, str(project_root))
```

