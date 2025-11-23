@echo off
echo ==========================================
echo Configuration des variables d'environnement
echo ==========================================
echo.

REM Modifiez ces valeurs selon vos besoins
set WANDB_API_KEY=your_wandb_api_key_here
set FLASK_LOG=flask.log
set PORT=5000

echo Variables configurees:
echo   WANDB_API_KEY=%WANDB_API_KEY%
echo   FLASK_LOG=%FLASK_LOG%
echo   PORT=%PORT%
echo.
echo IMPORTANT: Ces variables sont configurees pour cette session seulement
echo.
echo Pour utiliser ces variables:
echo   1. Executez: call set_env.bat
echo   2. Puis executez: start_server.bat
echo.
pause