# Script PowerShell pour démarrer le service Flask
# Usage: .\start_flask.ps1

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  NHL xG Tracker - Service Flask" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# 1. Vérifier que l'environnement virtuel existe
$venvPath = "D:\Bureau\project-datascience-python\ift6758-venv\Scripts\Activate.ps1"
if (-not (Test-Path $venvPath)) {
    Write-Host "[ERREUR] Environnement virtuel non trouvé!" -ForegroundColor Red
    Write-Host "   Chemin attendu: $venvPath" -ForegroundColor Yellow
    Write-Host "   Créez d'abord l'environnement virtuel avec: python -m venv ift6758-venv" -ForegroundColor Yellow
    exit 1
}

# 2. Charger les variables d'environnement depuis .env
$envFile = "D:\Bureau\project-datascience-python\.env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$' -and -not $_.StartsWith('#')) {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim()
            if ($value) {
                Set-Item -Path "env:$name" -Value $value
                Write-Host "[OK] Variable $name chargée depuis .env" -ForegroundColor Green
            }
        }
    }
}

# 3. Vérifier la clé API Wandb
if (-not $env:WANDB_API_KEY) {
    Write-Host "[ATTENTION] WANDB_API_KEY n'est pas définie!" -ForegroundColor Yellow
    Write-Host "   Le service ne pourra pas télécharger les modèles depuis Wandb" -ForegroundColor Yellow
    Write-Host "   Option 1: Définir dans .env à la racine du projet" -ForegroundColor Yellow
    Write-Host "   Option 2: Définir avec `$env:WANDB_API_KEY='votre_cle'" -ForegroundColor Yellow
    Write-Host ""
    $response = Read-Host "Voulez-vous entrer votre clé API maintenant? (o/n)"
    
    if ($response -eq 'o' -or $response -eq 'O') {
        $apiKey = Read-Host "Entrez votre clé API Wandb" -AsSecureString
        $env:WANDB_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
            [Runtime.InteropServices.Marshal]::SecureStringToBSTR($apiKey)
        )
        Write-Host "[OK] Clé API définie pour cette session" -ForegroundColor Green
    } else {
        Write-Host "[ERREUR] Le service ne pourra pas fonctionner correctement sans Wandb" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "Démarrage du service Flask..." -ForegroundColor Green
Write-Host ""

# 3. Activer l'environnement virtuel et démarrer Flask
& $venvPath
Set-Location "D:\Bureau\project-datascience-python\ift6758\backend"

Write-Host "[INFO] Lancement du service Flask..." -ForegroundColor Cyan
Write-Host "   URL: http://127.0.0.1:5000" -ForegroundColor Gray
Write-Host "   Logs: http://127.0.0.1:5000/logs" -ForegroundColor Gray
Write-Host ""
Write-Host "[INFO] Le service est maintenant en écoute..." -ForegroundColor Green
Write-Host "   Appuyez sur Ctrl+C pour arrêter" -ForegroundColor Gray
Write-Host ""

# Essayer waitress d'abord, sinon utiliser Flask en mode dev
if (Get-Command "waitress-serve" -ErrorAction SilentlyContinue) {
    waitress-serve --listen=127.0.0.1:5000 app:app
} elseif (python -c "import waitress" 2>$null) {
    python -m waitress --listen=127.0.0.1:5000 app:app
} else {
    Write-Host "[ATTENTION] Waitress non installé, utilisation de Flask en mode développement" -ForegroundColor Yellow
    $env:FLASK_APP="app.py"
    $env:FLASK_ENV="development"
    python -m flask run --host=127.0.0.1 --port=5000
}
