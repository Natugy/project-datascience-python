"""
Script de test pour l'application Streamlit
Permet de tester les composants individuellement sans lancer Streamlit
"""

import sys
from pathlib import Path

# Ajouter le chemin du module ift6758 au sys.path
# Le fichier est dans milestone_3/streamlit/
project_root = Path(__file__).parent.parent  # Remonter à milestone_3
sys.path.insert(0, str(project_root))

from serving_client import ServingClient
from game_client import GameClient


def test_serving_client():
    """Test du ServingClient"""
    print("=" * 60)
    print("TEST 1: ServingClient")
    print("=" * 60)
    
    try:
        # Créer le client
        client = ServingClient(ip="127.0.0.1", port=5000)
        print("[OK] ServingClient créé avec succès")
        
        # Tester /logs
        print("\nTest de /logs...")
        logs = client.logs()
        print("[OK] Logs récupérés")
        print(f"Extrait des logs: {str(logs)[:200]}...")
        
        # Tester /download_registry_model
        print("\nTest de /download_registry_model...")
        response = client.download_registry_model(
            workspace="qi-li-1-universit-de-montr-al",
            model="logreg-distance",
            version="v0"
        )
        print("[OK] Modèle téléchargé")
        print(f"Réponse: {response}")
        
        return True
        
    except Exception as e:
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_game_client():
    """Test du GameClient"""
    print("\n" + "=" * 60)
    print("TEST 2: GameClient")
    print("=" * 60)
    
    try:
        # Créer les clients
        serving_client = ServingClient(ip="127.0.0.1", port=5000)
        game_client = GameClient(serving_client=serving_client)
        print("[OK] GameClient créé avec succès")
        
        # Tester fetch_game
        print("\nTest de fetch_game...")
        game_id = "2021020329"
        game_data = game_client.fetch_game(game_id)
        print(f"[OK] Données du match {game_id} récupérées")
        
        # Tester get_game_metadata
        print("\nTest de get_game_metadata...")
        metadata = game_client.get_game_metadata()
        print("[OK] Métadonnées récupérées:")
        print(f"  - Domicile: {metadata['home_team']} ({metadata['home_score']})")
        print(f"  - Visiteur: {metadata['away_team']} ({metadata['away_score']})")
        print(f"  - Période: {metadata['period']}")
        print(f"  - Temps restant: {metadata['time_remaining']}")
        
        # Tester get_new_shots_for_prediction
        print("\nTest de get_new_shots_for_prediction...")
        df = game_client.get_new_shots_for_prediction()
        print(f"[OK] {len(df)} événements de tir récupérés")
        if not df.empty:
            print(f"Colonnes: {df.columns.tolist()}")
            print(f"Premières lignes:\n{df.head()}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_workflow():
    """Test du workflow complet"""
    print("\n" + "=" * 60)
    print("TEST 3: Workflow Complet")
    print("=" * 60)
    
    try:
        # 1. Créer les clients
        print("1. Création des clients...")
        serving_client = ServingClient(ip="127.0.0.1", port=5000)
        game_client = GameClient(serving_client=serving_client)
        print("[OK] Clients créés")
        
        # 2. Charger un modèle
        print("\n2. Chargement du modèle...")
        response = serving_client.download_registry_model(
            workspace="qi-li-1-universit-de-montr-al",
            model="logreg-distance",
            version="v0"
        )
        print(f"[OK] Modèle chargé: {response}")
        
        # 3. Ping un match
        print("\n3. Ping du match...")
        game_id = "2021020329"
        metadata, df_pred = game_client.ping_game(game_id)
        print(f"[OK] Match pingé: {metadata['home_team']} vs {metadata['away_team']}")
        
        if df_pred is not None and not df_pred.empty:
            print(f"[OK] {len(df_pred)} prédictions reçues")
            print(f"Colonnes: {df_pred.columns.tolist()}")
            print(f"\nPremières prédictions:")
            print(df_pred.head())
            
            # Calculer les xG par équipe
            if "teamId" in df_pred.columns:
                home_id = game_client.cached_game_data["homeTeam"]["id"]
                away_id = game_client.cached_game_data["awayTeam"]["id"]
                
                home_xg = df_pred[df_pred["teamId"] == home_id]["model_output"].sum()
                away_xg = df_pred[df_pred["teamId"] == away_id]["model_output"].sum()
                
                print(f"\nExpected Goals:")
                print(f"  {metadata['home_team']}: {home_xg:.2f} xG (score: {metadata['home_score']})")
                print(f"  {metadata['away_team']}: {away_xg:.2f} xG (score: {metadata['away_score']})")
        else:
            print("[INFO] Aucune nouvelle prédiction (match probablement déjà traité)")
        
        # 4. Second ping (devrait ne trouver aucun nouvel événement)
        print("\n4. Second ping (test du filtrage)...")
        metadata2, df_pred2 = game_client.ping_game(game_id)
        if df_pred2 is None or df_pred2.empty:
            print("[OK] Aucun nouvel événement détecté (filtrage fonctionne)")
        else:
            print(f"[WARNING] {len(df_pred2)} nouveaux événements (inattendu pour un match terminé)")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Exécuter tous les tests"""
    print("\n" + "=" * 60)
    print("TESTS DE L'APPLICATION STREAMLIT")
    print("=" * 60)
    print("\n[ATTENTION] Assurez-vous que le service Flask est démarré sur localhost:5000")
    print("[ATTENTION] Assurez-vous que WANDB_API_KEY est configuré\n")
    
    input("Appuyez sur Entrée pour continuer...")
    
    results = []
    
    # Test 1: ServingClient
    results.append(("ServingClient", test_serving_client()))
    
    # Test 2: GameClient
    results.append(("GameClient", test_game_client()))
    
    # Test 3: Workflow complet
    results.append(("Workflow Complet", test_full_workflow()))
    
    # Résumé
    print("\n" + "=" * 60)
    print("RÉSUMÉ DES TESTS")
    print("=" * 60)
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} - {name}")
    
    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n[SUCCESS] Tous les tests ont réussi!")
    else:
        print("\n[WARNING] Certains tests ont échoué. Vérifiez les logs ci-dessus.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
