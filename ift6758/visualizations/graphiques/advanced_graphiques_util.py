from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache
from typing import Dict, Tuple, Any, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
import base64
# =========================
# Utilitaires coordonnées
# =========================
def corriger_orientation_patinoire(x_coords, y_coords) -> Tuple[np.ndarray, np.ndarray]:
    """Oriente toutes les coordonnées vers la zone offensive (x >= 0)."""
    x = np.asarray(x_coords, dtype=float)
    y = np.asarray(y_coords, dtype=float)
    m = x < 0
    f = np.where(m, -1.0, 1.0)
    return x * f, y * f


# =========================
# Chargement CSV saison
# =========================
def charger_saison_csv(saison: str, dossier_csv: str) -> pd.DataFrame:
    """
    Charge le CSV `dossier_csv/<saison>.csv`.
    Colonnes attendues (au moins) : xCoord, yCoord, idGame, teamId ou teamAbbr.
    """
    chemin = os.path.join(dossier_csv, f"{saison}.csv")
    if not os.path.exists(chemin):
        raise FileNotFoundError(f"CSV introuvable: {chemin}")

    df = pd.read_csv(chemin, low_memory=False)
    # Nettoyages légers & types
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    # Harmonise quelques types utiles
    if "idGame" in df.columns:
        df["idGame"] = df["idGame"].astype(str)
    for c in ("xCoord", "yCoord"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Vérifs minimales
    required_any_team = ("teamAbbr" in df.columns) or ("teamId" in df.columns)
    required_coords = {"xCoord", "yCoord"}.issubset(df.columns)
    if not required_any_team or not required_coords or "idGame" not in df.columns:
        cols = ", ".join(df.columns)
        raise KeyError(
            "Colonnes minimales manquantes. Requis: xCoord, yCoord, idGame, et teamAbbr OU teamId.\n"
            f"Colonnes trouvées: {cols}"
        )
    return df


# =========================================
# Préparation meta & edges
# =========================================
def preparer_donnees_tirs_saison_csv(
    saison: str,
    dossier_csv: str,
    image_patinoire: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Charge le CSV de la saison, et prépare df_prep + meta.
    """
    d = charger_saison_csv(saison, dossier_csv)

    # Paramètres patinoire (ft)
    largeur_case = 2.0
    largeur_patinoire = 80# A ajuster
    hauteur_patinoire = 100.0

    x_edges = np.arange(-hauteur_patinoire/2, largeur_patinoire/2 + largeur_case, largeur_case, dtype=float)
    y_edges = np.arange(-hauteur_patinoire/2, hauteur_patinoire/2 + largeur_case, largeur_case, dtype=float)

    # Liste d’équipes (ID si dispo sinon abbr)
    if "teamId" in d.columns:
        equipes = sorted(pd.to_numeric(d["teamId"], errors="coerce").dropna().unique().tolist())
    else:
        equipes = sorted(d["teamAbbr"].dropna().unique().tolist())

    meta = {
        "image_patinoire": image_patinoire,
        "largeur_case": float(largeur_case),
        "largeur_patinoire": float(largeur_patinoire),
        "hauteur_patinoire": float(hauteur_patinoire),
        "liste_saisons": [saison[:4]],
        "liste_equipes": equipes,
        "edges": (x_edges, y_edges),
        "baseline_cache": {},  # mémo local optionnel
    }
    return d, meta


# ==================================
# Baseline ligue (mémorisée)
# ==================================
@lru_cache(maxsize=64)
def _baseline_saison_cache_key(saison_key: str, x_edges_tuple: tuple, y_edges_tuple: tuple, df_hash: int):
    """Clé de cache immuable pour baseline LRU (utilitaire interne)."""
    return (saison_key, x_edges_tuple, y_edges_tuple, df_hash)

def _fast_df_hash(df: pd.DataFrame, cols=("idGame", "xCoord", "yCoord")) -> int:
    """Petit hash déterministe (non crypto) pour invalider proprement le cache si le CSV change."""
    # On évite un coût trop élevé : on ne hash que quelques colonnes &  prime simple
    h = 0
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.util.hash_pandas_object(df[c], index=False).values
        if len(s):
            h = (h * 1000003 + int(s.sum())) & 0xFFFFFFFF
    return h

def _baseline_saison(df_prep: pd.DataFrame, meta: Dict[str, Any], saison: str):
    """
    Calcule (ou récupère) l’histogramme 2D « ligue » pour une saison donnée.
    Normalisation ~ (# matchs uniques * 2) pour approx heures/équipe constantes.
    """
    # Cache local dans meta (rapide) prioritaire
    cache_local = meta.get("baseline_cache", {})
    if saison in cache_local:
        return cache_local[saison]

    x_edges, y_edges = meta["edges"]
    mask = df_prep["idGame"].str.startswith(saison[:4])
    saison_df = df_prep.loc[mask, ["xCoord", "yCoord", "idGame"]].copy()

    if saison_df.empty:
        hist0 = np.zeros((x_edges.size - 1, y_edges.size - 1), float)
        out = (hist0, x_edges, y_edges, 0)
        cache_local[saison] = out
        meta["baseline_cache"] = cache_local
        return out

    # Hash pour LRU global (optionnel, protège d’un DF différent)
    dfh = _fast_df_hash(saison_df)
    lru_key = _baseline_saison_cache_key(
        saison[:4],
        tuple(np.round(x_edges, 6)),
        tuple(np.round(y_edges, 6)),
        dfh
    )

    @lru_cache(maxsize=64)
    def _compute_baseline(_key):
        xs = saison_df["xCoord"].to_numpy()
        ys = saison_df["yCoord"].to_numpy()
        ok = np.isfinite(xs) & np.isfinite(ys)
        if not np.any(ok):
            hist0 = np.zeros((x_edges.size - 1, y_edges.size - 1), float)
            return (hist0, x_edges, y_edges, 0)
        xs, ys = corriger_orientation_patinoire(xs[ok], ys[ok])
        hist, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges])
        n_games = saison_df.loc[ok, "idGame"].nunique()
        denom = max(n_games * 2, 1)
        hist_norm = hist / denom
        return (hist_norm, x_edges, y_edges, int(n_games))

    out = _compute_baseline(lru_key)
    cache_local[saison] = out
    meta["baseline_cache"] = cache_local
    return out


# ==================================
# Tracé équipe vs ligue (saison)
# ==================================
def _pick_team_mask(df: pd.DataFrame, team: Union[str, int]) -> np.ndarray:
    """
    Retourne un masque bool pour la colonne d’équipe pertinente.
    - Si team est str -> on utilise teamAbbr
    - Si team est int/float -> on utilise teamId
    """
    if isinstance(team, str):
        if "teamAbbr" not in df.columns:
            raise KeyError("teamAbbr absent du CSV : passe un teamId (int) ou ajoute cette colonne à l’extraction.")
        return (df["teamAbbr"].astype(str) == team)
    else:
        if "teamId" not in df.columns:
            raise KeyError("teamId absent du CSV : passe un teamAbbr (str) ou ajoute cette colonne à l’extraction.")
        # Tolère int/float/str
        team_num = pd.to_numeric(pd.Series([team]), errors="coerce").iloc[0]
        return (pd.to_numeric(df["teamId"], errors="coerce") == team_num)

def plot_team_season(
    df_prep: pd.DataFrame,
    meta: Dict[str, Any],
    team_name: Union[str, int],
    season: str,
    *,
    sigma_lissage: float = 1.5,
    afficher: bool = True,
):
    """Trace la différence de taux de tir (équipe – ligue) pour une saison."""
    hist_saison, x_edges, y_edges, n_games_season = _baseline_saison(df_prep, meta, season)

    ms = df_prep["idGame"].str.startswith(season[:4])
    mt = _pick_team_mask(df_prep, team_name)
    team_df = df_prep.loc[ms & mt, ["xCoord", "yCoord", "idGame"]]

    if team_df.empty:
        print(f"Aucune donnée pour {team_name} en {season}")
        return None, None, {}

    xt = team_df["xCoord"].to_numpy()
    yt = team_df["yCoord"].to_numpy()
    ok = np.isfinite(xt) & np.isfinite(yt)
    if not np.any(ok):
        print(f"Aucune coordonnée valide pour {team_name} en {season}")
        return None, None, {}

    xt, yt = corriger_orientation_patinoire(xt[ok], yt[ok])
    hist_team, _, _ = np.histogram2d(xt, yt, bins=[x_edges, y_edges])

    n_games_team = team_df.loc[ok, "idGame"].nunique()
    hist_team_norm = hist_team / max(int(n_games_team), 1)

    # Différence puis lissage
    diff = hist_team_norm - hist_saison
    if sigma_lissage and sigma_lissage > 0:
        diff = gaussian_filter(diff, sigma=sigma_lissage)

    # Échelle robuste et symétrique (évite que quelques outliers écrasent la palette)
    p98 = float(np.nanpercentile(np.abs(diff), 98)) if np.isfinite(diff).any() else 1e-6
    vmax = max(p98, 1e-6)
    vmin = -vmax
    levels = np.linspace(vmin, vmax, 21)  # 20 bandes régulières

    # ---------- FIGURE ----------
    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    # Patinoire en fond
    ax.imshow(
        meta["image_patinoire"],
        extent=[-meta["largeur_patinoire"]/2, meta["largeur_patinoire"]/2,
                -meta["hauteur_patinoire"]/2,  meta["hauteur_patinoire"]/2],
        aspect="auto",
        alpha=0.7,             
        interpolation="bilinear",
        zorder=0,
    )

    # Heatmap principale en verticale
    X, Y = np.meshgrid(y_edges[:-1], x_edges[:-1], indexing="xy")
    cs = ax.contourf(
        X, Y, diff,
        levels=levels,
        cmap="RdBu_r",
        extend="both",
        antialiased=True,
        zorder=1,
    )

    # Isolignes fines pour accentuer le relief
    ax.contour(
        X, Y, diff,
        levels=np.linspace(vmin, vmax, 11),
        colors="k",
        linewidths=0.4,
        alpha=0.25,
        zorder=2,
    )

    # Barre de couleurs lisible et symétrique
    tick_vals = np.linspace(vmin, vmax, 9)
    cbar = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks(tick_vals)
    cbar.ax.tick_params(labelsize=9)

    # Axes & titre
    ax.set_title(f"Plan de tirs de l'équipe {team_name} pour la saison {season}\n")
    ax.set_xlabel("Distance latérale (ft)")
    ax.set_ylabel("Distance depuis le centre (ft)")
    ax.set_xlim([-meta["largeur_patinoire"]/2, meta["largeur_patinoire"]/2])
    ax.set_ylim([-meta["hauteur_patinoire"]/2,  meta["hauteur_patinoire"]/2])
    ax.set_aspect("equal", adjustable="box")

    # Grille légère (améliore la lecture sans parasiter)
    ax.grid(True, color="0.9", linewidth=0.6, alpha=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.5")

    fig.tight_layout()

    if afficher:
        plt.show()

    return fig, ax

