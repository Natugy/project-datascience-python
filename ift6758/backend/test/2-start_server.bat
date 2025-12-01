
@echo off
title Flask App - Waitress Server

echo ==========================================
echo Demarrage du serveur Flask avec Waitress
echo ==========================================
echo.

REM Configuration du port
set PORT=5000
if not "%1"=="" set PORT=%1

REM Variables d'environnement
if "%WANDB_API_KEY%"=="" (
    echo ATTENTION: WANDB_API_KEY n'est pas definie
    echo Vous pouvez la definir avec: set WANDB_API_KEY=votre_cle
    echo.
)

if "%FLASK_LOG%"=="" (
    set FLASK_LOG=flask.log
    echo FLASK_LOG non defini, utilisation de: flask.log
)

echo Configuration:
echo   Port: %PORT%
echo   Log file: %FLASK_LOG%
if not "%WANDB_API_KEY%"=="" (
    echo   WANDB_API_KEY: Configuree
) else (
    echo   WANDB_API_KEY: Non configuree
)
echo.

REM Aller dans le repertoire backend (parent du repertoire test)
cd /d "%~dp0.."

REM Verifier que app.py existe
if not exist "app.py" (
    echo ERREUR: app.py introuvable dans le repertoire backend
    echo Repertoire actuel: %CD%
    pause
    exit /b 1
)

REM Configurer PYTHONPATH pour trouver le module ift6758
REM Remonter de 2 niveaux: backend -> ift6758 -> project-datascience-python
set BACKEND_DIR=%CD%
cd /d "%~dp0..\.."
set PROJECT_ROOT=%CD%

REM Revenir dans backend
cd /d "%BACKEND_DIR%"

REM Ajouter le repertoire racine du projet au PYTHONPATH
set PYTHONPATH=%PROJECT_ROOT%;%PYTHONPATH%

echo PYTHONPATH configure: %PROJECT_ROOT%
echo Repertoire de travail: %CD%
echo.

echo Demarrage du serveur Waitress...
echo.
echo ==========================================
echo Serveur accessible sur: http://localhost:%PORT%
echo Appuyez sur Ctrl+C pour arreter le serveur
echo ==========================================
echo.

REM Demarrer Waitress depuis le repertoire backend
waitress-serve --listen=0.0.0.0:%PORT% --threads=4 --channel-timeout=300 app:app

pause