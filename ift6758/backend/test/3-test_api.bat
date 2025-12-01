@echo off
echo ==========================================
echo Test de l'API Flask avec curl
echo ==========================================
echo.

REM Configuration
set PORT=5000
if not "%1"=="" set PORT=%1
set BASE_URL=http://localhost:%PORT%

echo Port: %PORT%
echo Base URL: %BASE_URL%
echo.

REM Verifier que curl est disponible
curl --version >nul 2>&1
if errorlevel 1 (
    echo ERREUR: curl n'est pas installe ou pas dans le PATH
    echo.
    echo Solutions:
    echo 1. Telechargez curl depuis: https://curl.se/windows/
    echo 2. Ajoutez curl au PATH de Windows
    echo.
    pause
    exit /b 1
)

echo ==========================================
echo Test 1/5: GET /logs
echo ==========================================
curl -s -w "\nHTTP Status: %%{http_code}\n" "%BASE_URL%/logs"
echo.
echo.
pause

echo ==========================================
echo Test 2/5: POST /download_registry_model (modele valide)
echo ==========================================
curl -s -X POST "%BASE_URL%/download_registry_model" ^
  -H "Content-Type: application/json" ^
  -d "{\"model\": \"logreg-distance\", \"version\": \"v0\"}"
echo.
echo.
pause

echo ==========================================
echo Test 3/5: POST /download_registry_model (modele invalide)
echo ==========================================
curl -s -X POST "%BASE_URL%/download_registry_model" ^
  -H "Content-Type: application/json" ^
  -d "{\"model\": \"invalid-model\"}"
echo.
echo.
pause
echo ==========================================
echo Test 5/5: Flux complet - Download puis Predict
echo ==========================================
echo.
echo Etape 1: Telechargement du modele...
curl -s -X POST "%BASE_URL%/download_registry_model" ^
  -H "Content-Type: application/json" ^
  -d "{\"model\": \"logreg-distance\"}"
echo.
echo.
echo Etape 2: Prediction avec distance_net...
curl -s -X POST "%BASE_URL%/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"distance_net\": 10.5}"
echo.
echo.
echo ==========================================
echo Test 2/6: logreg-distance-angle
echo ==========================================
echo.
echo Telechargement du modele distance-angle...
curl -s -X POST "%BASE_URL%/download_registry_model" ^
  -H "Content-Type: application/json" ^
  -d "{\"model\": \"logreg-distance-angle\"}"
echo.
echo.
echo Prediction avec distance_net et angle_net...
curl -s -X POST "%BASE_URL%/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"distance_net\": 10.5, \"angle_net\": 0.3}"
echo.
echo.
echo ==========================================
echo Tests termines!
echo ==========================================
pause