"""
Point d'entrée exécutable pour l'explorateur interactif NHL.
Permet de lancer via : python -m ift6758.visualizations.debogage_interactif
"""
import sys, subprocess
from pathlib import Path

APP = Path(__file__).with_name("app_event_explorer.py")

def main():
    cmd = [sys.executable, "-m", "streamlit", "run", str(APP)]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("Erreur lors du lancement de l'explorateur :", e)

if __name__ == "__main__":
    main()
