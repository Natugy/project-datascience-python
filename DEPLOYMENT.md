# Déploiement sur Streamlit Community Cloud

## Étapes de déploiement:

### 1. Préparer le repository GitHub
- ✅ Votre code est déjà sur GitHub: `https://github.com/Natugy/project-datascience-python`
- ✅ Assurez-vous que la branche `main` est à jour

### 2. Créer un compte Streamlit Cloud
1. Allez sur: https://share.streamlit.io/
2. Cliquez sur "Sign in" puis "Continue with GitHub"
3. Autorisez Streamlit à accéder à vos repositories GitHub

### 3. Déployer l'application
1. Cliquez sur "New app" dans le dashboard Streamlit Cloud
2. Remplissez les informations:
   - **Repository:** `Natugy/project-datascience-python`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
3. Cliquez sur "Advanced settings" et ajoutez:
   - **Python version:** `3.9`
   
4. Ajoutez les secrets (variables d'environnement):
   ```toml
   WANDB_API_KEY = "13a4f31490980ca10265e2cbf4d46f26ba7a9a7b"
   ```

5. Cliquez sur "Deploy!"

### 4. Configuration des secrets
Dans Streamlit Cloud, allez dans "Settings" > "Secrets" et ajoutez:
```toml
WANDB_API_KEY = "13a4f31490980ca10265e2cbf4d46f26ba7a9a7b"
```

### 5. Accéder à votre application
Une fois déployée, votre app sera accessible à:
`https://natugy-project-datascience-python-streamlit-app-xxxxx.streamlit.app`

## Notes importantes:

⚠️ **Backend Flask non disponible sur Streamlit Cloud**
- Streamlit Cloud ne peut héberger que l'application Streamlit
- Le backend Flask (`serving`) ne sera pas disponible
- Vous devrez soit:
  1. Héberger le backend Flask séparément (Heroku, Railway, etc.)
  2. Modifier l'app pour charger les modèles directement dans Streamlit (sans Flask)

⚠️ **Données volumineuses**
- Les fichiers dans `data/raw/` sont très volumineux (>100MB)
- Cela peut ralentir le déploiement
- Considérez exclure ces fichiers si non essentiels

## Alternative: Backend intégré
Pour simplifier, vous pouvez modifier l'app pour qu'elle charge les modèles directement depuis Wandb sans passer par Flask.
