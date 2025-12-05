# Script PowerShell pour démarrer l'application Streamlit en mode développement
# Usage: .\start_local.ps1

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  NHL xG Tracker - Démarrage Local" -ForegroundColor Cyan
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
        Write-Host "[ATTENTION] L'application continuera sans Wandb" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Démarrage de l'application..." -ForegroundColor Green
Write-Host ""

# 4. Activer l'environnement virtuel et démarrer Streamlit
& $venvPath
Set-Location "D:\Bureau\project-datascience-python\ift6758\milestone_3\streamlit"

# Définir PYTHONPATH pour inclure le dossier milestone_3
$env:PYTHONPATH = "D:\Bureau\project-datascience-python\ift6758\milestone_3"

Write-Host "[INFO] Lancement de Streamlit..." -ForegroundColor Cyan
Write-Host "   L'application s'ouvrira automatiquement dans votre navigateur" -ForegroundColor Gray
Write-Host "   URL: http://localhost:8501" -ForegroundColor Gray
Write-Host "   Config: .streamlit_config/config.toml" -ForegroundColor Gray
Write-Host ""
Write-Host "[ATTENTION] N'oubliez pas de démarrer le service Flask dans un autre terminal!" -ForegroundColor Yellow
Write-Host "   Commande: .\start_flask.ps1 (dans ce même dossier)" -ForegroundColor Gray
Write-Host ""

# Définir le dossier de configuration Streamlit
$env:STREAMLIT_CONFIG_DIR = ".streamlit_config"

streamlit run streamlit_app.py
