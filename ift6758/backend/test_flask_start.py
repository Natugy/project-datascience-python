"""
Script de test pour vérifier que Flask peut démarrer
"""
import sys
from pathlib import Path

# Ajouter le chemin parent pour importer app
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Importing Flask app...")
from backend import app

print("Flask app imported successfully!")
print(f"App name: {app.app.name}")
print(f"App debug: {app.app.debug}")

print("\nStarting Flask server...")
try:
    app.app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
