# Données NHL

## Raw Data (data/raw/)
- Données brutes des saisons NHL 2016-2024
- Format: JSON (API NHL) et CSV (convertis)
- Source: NHL Stats API

## Processed Data (data/processed/)
- `train_data.csv`: Saisons 2016-2020 (80%) - 193,227 tirs
- `val_data.csv`: Saisons 2016-2020 (20%) - 48,307 tirs  
- `test_data.csv`: Saison 2020-2021 - 57,734 tirs

## Features
- 18 features après encodage (11 numériques + 7 shot_type one-hot)
- Target: `is_goal` (0 = non-but, 1 = but)
- Taux de buts: ~9.5% (déséquilibré)
