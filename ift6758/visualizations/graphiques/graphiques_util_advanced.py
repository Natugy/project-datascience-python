'''
Use python -m ift6758.visualizations.graphiques.graphiques_util_advanced to test.

'''
from __future__ import annotations

import os, io, json, glob, base64
from functools import lru_cache
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go

# Matplotlib uniquement pour le fond patinoire / PNG statiques
try:
    import matplotlib.pyplot as plt
except Exception: 
    plt = None

# Tentative d’import du dessin de patinoire (sinon fond désactivé)
try:
    from ift6758.visualizations.debogage_interactif.patinoire import draw_rink
except Exception:
    draw_rink = None  # fond indisponible

# ---------------------------------------------------------------------------
# Constantes patinoire & grille
# ---------------------------------------------------------------------------
X_MAX, Y_MAX = 100.0, 42.5          # demi-patinoire offensive (x>=0)
NX_DEFAULT, NY_DEFAULT = 50, 34     # résolution par défaut
SIGMA_DEFAULT = 1.2                 # lissage gaussien
SHOT_TYPES = {"shot-on-goal", "goal"}

# Palette divergente inspirée HockeyViz (bleu ← 0 → rouge)
HOCKEYVIZ_CS = [
    [0.00, "#08306B"],
    [0.08, "#08519C"],
    [0.18, "#2171B5"],
    [0.32, "#6BAED6"],
    [0.45, "#C6DBEF"],
    [0.50, "#F7F7F7"],  # blanc au centre
    [0.55, "#FEE0D2"],
    [0.68, "#FCBBA1"],
    [0.82, "#FB6A4A"],
    [0.92, "#EF3B2C"],
    [1.00, "#CB181D"],
]

# ---------------------------------------------------------------------------
# Utilitaires grille
# ---------------------------------------------------------------------------
def _bin_edges(nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    xedges = np.linspace(0, X_MAX, nx + 1)
    yedges = np.linspace(-Y_MAX, Y_MAX, ny + 1)
    return xedges, yedges

def _grid_centers(nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    xedges, yedges = _bin_edges(nx, ny)
    return 0.5 * (xedges[:-1] + xedges[1:]), 0.5 * (yedges[:-1] + yedges[1:])

def _hist2d(xx: np.ndarray, yy: np.ndarray, nx: int, ny: int) -> np.ndarray:
    xedges, yedges = _bin_edges(nx, ny)
    H, _, _ = np.histogram2d(
        xx, yy, bins=[xedges, yedges],
        range=[[0, X_MAX], [-Y_MAX, Y_MAX]],
    )
    return H

# ---------------------------------------------------------------------------
# Orientation / normalisation (x, y)
# ---------------------------------------------------------------------------
def _home_attacks_right(home_def_side: Optional[str]) -> bool:
    # homeTeamDefendingSide == 'left' -> home défend à gauche => attaque à droite
    return (home_def_side or "").lower() == "left"

def _normalize_xy(
    x: Optional[float], y: Optional[float],
    shooter_is_home: bool, home_def_side: Optional[str],
) -> Optional[Tuple[float, float]]:
    if x is None or y is None:
        return None
    x, y = float(x), float(y)
    shooting_right = _home_attacks_right(home_def_side) if shooter_is_home else not _home_attacks_right(home_def_side)
    if not shooting_right:
        x, y = -x, -y
    if x < 0:
        return None
    return x, y

# ---------------------------------------------------------------------------
# Lecture / itération des matchs
# ---------------------------------------------------------------------------
def _bundle_path(season: str, dest_folder: str) -> str:
    return os.path.join(dest_folder, f"{season}.json")

@lru_cache(maxsize=64)
def _load_bundle(season: str, dest_folder: str) -> List[Dict]:
    path = _bundle_path(season, dest_folder)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    return []

def _iter_games_from_season(season: str, dest_folder: str) -> Iterable[Dict]:
    """Privilégie le bundle <season>.json, puis les fichiers unitaires YYYY0[23]*.json, sans doublons."""
    seen_ids: set = set()

    # bundle
    for g in _load_bundle(season, dest_folder):
        if isinstance(g, dict) and "plays" in g:
            gid = g.get("id") or g.get("gameId") or g.get("gamePk")
            if gid is not None:
                if gid in seen_ids:
                    continue
                seen_ids.add(gid)
            yield g

    # fallback fichiers unitaires
    pattern = os.path.join(dest_folder, f"{season[:4]}0[23]*.json")
    for path in glob.glob(pattern):
        try:
            with open(path, "r", encoding="utf-8") as f:
                g = json.load(f)
            if isinstance(g, dict) and "plays" in g:
                gid = g.get("id") or g.get("gameId") or g.get("gamePk")
                if gid is not None and gid in seen_ids:
                    continue
                if gid is not None:
                    seen_ids.add(gid)
                yield g
        except Exception:
            continue

def _teams_from(game: Dict) -> Tuple[Optional[int], Optional[int], Optional[str], Optional[str]]:
    H = game.get("homeTeam") or game.get("home") or {}
    A = game.get("awayTeam") or game.get("away") or {}
    def _id(T: Dict)   -> Optional[int]:  return T.get("id") or T.get("teamId")
    def _abbr(T: Dict) -> Optional[str]:  return T.get("abbrev") or T.get("triCode") or T.get("shortName")
    return _id(H), _id(A), _abbr(H), _abbr(A)

# ---------------------------------------------------------------------------
# Agrégation (ligue & équipes)
# ---------------------------------------------------------------------------
def collect_shots_one_season(
    season: str, dest_folder: str, *,
    sigma: float = SIGMA_DEFAULT, nx: int = NX_DEFAULT, ny: int = NY_DEFAULT,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str]]:
    nx = int(nx); ny = int(ny); sigma = float(sigma)

    league_hist = np.zeros((nx, ny), float)
    team_hist   = defaultdict(lambda: np.zeros((nx, ny), float))
    league_games = 0
    team_games   = defaultdict(int)

    for game in _iter_games_from_season(season, dest_folder):
        plays = game.get("plays") or []
        home_id, away_id, H, A = _teams_from(game)
        if not (home_id and away_id and H and A):
            continue

        bucket = {H: [], A: []}
        for ev in plays:
            if ev.get("typeDescKey") not in SHOT_TYPES:
                continue
            det = ev.get("details") or {}
            x, y = det.get("xCoord"), det.get("yCoord")
            if x is None or y is None:
                continue
            shooter_is_home = (det.get("eventOwnerTeamId") == home_id)
            xy = _normalize_xy(x, y, shooter_is_home, ev.get("homeTeamDefendingSide"))
            if xy is None:
                continue
            bucket[H if shooter_is_home else A].append(xy)

        # Ligue
        all_xy = [p for L in bucket.values() for p in L]
        if all_xy:
            xx = np.fromiter((p[0] for p in all_xy), float)
            yy = np.fromiter((p[1] for p in all_xy), float)
            league_hist += _hist2d(xx, yy, nx, ny)
        league_games += 1

        # Équipes
        for abbr, L in bucket.items():
            if L:
                xx = np.fromiter((p[0] for p in L), float)
                yy = np.fromiter((p[1] for p in L), float)
                team_hist[abbr] += _hist2d(xx, yy, nx, ny)
            team_games[abbr] += 1

    league_sm = gaussian_filter(league_hist, sigma=sigma)
    for k in list(team_hist.keys()):
        team_hist[k] = gaussian_filter(team_hist[k], sigma=sigma)

    if league_games > 0:
        league_sm /= league_games
    for k in list(team_hist.keys()):
        team_hist[k] /= max(1, team_games[k])

    league_map = league_sm.T
    team_maps  = {k: v.T for k, v in team_hist.items()}
    teams = sorted(team_maps.keys())
    return league_map, team_maps, teams

# ---------------------------------------------------------------------------
# Fond patinoire (image base64 pour Plotly)
# ---------------------------------------------------------------------------
def _rink_png_base64(away_label: str = "AWAY", home_label: str = "HOME",
                     width_px: int = 1200, height_px: int = 510) -> Optional[str]:
    if draw_rink is None or plt is None:
        return None
    fig, ax = plt.subplots(figsize=(width_px/100, height_px/100), dpi=100)
    ax.set_xlim(-100, 100); ax.set_ylim(-42.5, 42.5)
    draw_rink(ax, away_label=away_label, home_label=home_label)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")

# ---------------------------------------------------------------------------
# Figure interactive
# ---------------------------------------------------------------------------
def make_season_figure_enhanced(
    season: str, dest_folder: str, *,
    sigma: float = SIGMA_DEFAULT, nx: int = NX_DEFAULT, ny: int = NY_DEFAULT,
    show_rink: bool = True, rink_opacity: float = 0.35,
    use_contour: bool = True,
) -> Tuple[go.Figure, List[str]]:
    """
    Construit la figure interactive (différence de taux de tirs/heure, équipe − ligue).
    - use_contour=True : rendu « filled contour » (style HockeyViz)
    - use_contour=False: rendu heatmap
    """
    league_map, team_maps, teams = collect_shots_one_season(
        season, dest_folder, sigma=sigma, nx=nx, ny=ny
    )
    if not teams:
        raise RuntimeError(f"Aucune équipe trouvée pour la saison {season}")

    xcent, ycent = _grid_centers(nx, ny)
    first = teams[0]
    diff0 = team_maps[first] - league_map

    # Bornes symétriques limitées par percentile pour éviter les outliers
    amax = np.nanmax(np.abs(diff0))
    v98  = np.nanpercentile(np.abs(diff0), 98)
    vmax = float(max(1e-9, min(amax, v98)))

    if use_contour:
        base_trace = go.Contour(
            x=xcent, y=ycent, z=diff0,
            colorscale=HOCKEYVIZ_CS, zmid=0, zmin=-vmax, zmax=vmax,
            contours=dict(coloring="heatmap", showlines=False,
                          start=-vmax, end=vmax, size=vmax/9),
            colorbar=dict(title="Excess shots per hour",
                          ticks="outside", tick0=-1.0, dtick=0.2),
            opacity=0.88
        )
    else:
        base_trace = go.Heatmap(
            x=xcent, y=ycent, z=diff0,
            colorscale=HOCKEYVIZ_CS, zmid=0, zmin=-vmax, zmax=vmax,
            colorbar=dict(title="Excess shots per hour",
                          ticks="outside", tick0=-1.0, dtick=0.2),
            hovertemplate="x=%{x:.1f} ft<br>y=%{y:.1f} ft<br>Δ=%{z:.3f}<extra></extra>",
            opacity=0.82, zsmooth="best"
        )

    fig = go.Figure(base_trace)

    # Fond patinoire
    if show_rink:
        b64 = _rink_png_base64(away_label="AWAY", home_label="HOME")
        if b64 is not None:
            fig.add_layout_image(dict(
                source=f"data:image/png;base64,{b64}",
                xref="x", yref="y",
                x=0, y=Y_MAX, sizex=X_MAX, sizey=2*Y_MAX,
                sizing="stretch", opacity=rink_opacity, layer="below",
            ))

    # Menu d’équipes
    restyle_common = {
        "zmin": [-vmax], "zmax": [vmax], "zmid": [0],
        "colorscale": [HOCKEYVIZ_CS],
        "opacity": [0.88 if use_contour else 0.82],
    }
    buttons = [
        dict(
            label=abbr, method="restyle",
            args=[{**{"z": [(team_maps[abbr] - league_map).tolist()]}, **restyle_common}],
        )
        for abbr in teams
    ]

    fig.update_layout(
        title=f"{season} — Différence de taux de tirs/heure (équipe − ligue), zone offensive",
        updatemenus=[dict(type="dropdown", x=0.02, y=1.12, showactive=True, buttons=buttons)],
        template="plotly_white",
        margin=dict(l=30, r=30, t=60, b=40),
    )
    fig.update_yaxes(scaleanchor="x", range=[-Y_MAX, Y_MAX], title="Distance from centre of rink(ft)")
    fig.update_xaxes(range=[0, X_MAX], title="Distance from goal line (ft)")

    return fig, teams

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------
def write_html(fig: go.Figure, out_html: str) -> None:
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn", full_html=True)

def export_seasons_html(
    seasons: List[str], dest_folder: str, out_dir: str, *,
    sigma: float = SIGMA_DEFAULT, nx: int = NX_DEFAULT, ny: int = NY_DEFAULT,
    show_rink: bool = True, rink_opacity: float = 0.35,
    use_contour: bool = True,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for s in seasons:
        fig, _ = make_season_figure_enhanced(
            s, dest_folder, sigma=sigma, nx=nx, ny=ny,
            show_rink=show_rink, rink_opacity=rink_opacity,
            use_contour=use_contour
        )
        write_html(fig, os.path.join(out_dir, f"{s}_offence_map.html"))

# ---------------------------------------------------------------------------
# PNG statique pour une équipe
# ---------------------------------------------------------------------------
def save_team_map_png(
    season: str, dest_folder: str, *, team_abbr: str, out_png: str,
    sigma: float = SIGMA_DEFAULT, nx: int = NX_DEFAULT, ny: int = NY_DEFAULT,
    cmap: str = "RdBu", use_contour: bool = True
) -> None:
    """
    Sauvegarde un PNG statique (matplotlib) pour une équipe précise.
    use_contour=True applique des niveaux « filled-contour » pour un rendu proche HockeyViz.
    """
    if plt is None:
        raise RuntimeError("Matplotlib n'est pas disponible pour l'export PNG.")

    league_map, team_maps, teams = collect_shots_one_season(
        season, dest_folder, sigma=sigma, nx=nx, ny=ny
    )
    if team_abbr not in team_maps:
        raise ValueError(f"Équipe '{team_abbr}' introuvable pour {season} (trouvées: {', '.join(teams)})")

    diff = team_maps[team_abbr] - league_map
    amax = np.nanmax(np.abs(diff))
    v98  = np.nanpercentile(np.abs(diff), 98)
    vmax = float(max(1e-9, min(amax, v98)))
    xcent, ycent = _grid_centers(nx, ny)

    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=140)
    if draw_rink is not None:
        ax.set_xlim(-100, 100); ax.set_ylim(-42.5, 42.5)
        draw_rink(ax, away_label="AWAY", home_label="HOME")
        ax.set_xlim(0, 100)

    if use_contour:
        levels = np.linspace(-vmax, vmax, 19)
        cf = ax.contourf(
            xcent, ycent, diff, levels=levels, cmap="RdBu_r", vmin=-vmax, vmax=vmax, alpha=0.88
        )
        cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    else:
        im = ax.imshow(
            diff, extent=[xcent.min(), xcent.max(), ycent.min(), ycent.max()],
            origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax, alpha=0.85, aspect="auto"
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    cbar.set_label("Excess shots per hour")
    ax.set_title(f"{season} — {team_abbr} (offence vs ligue)")
    ax.set_xlabel("Distance depuis le centre (ft)")
    ax.set_ylabel("Distance latérale (ft)")

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Exécution rapide (optionnelle)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Adapte ces chemins à ton repo local
    DEST = r".\ressources"
    OUT  = r".\figures"
    seasons = ["20162017", "20172018", "20182019", "20192020", "20202021"]

    # Export HTML pour 5 saisons (contours + patinoire)
    export_seasons_html(
        seasons, DEST, OUT, nx=80, ny=54, sigma=1.6,
        show_rink=True, rink_opacity=0.35, use_contour=True
    )
    print("Export HTML terminé.")

    # Exemple PNG statiques
    save_team_map_png("20162017", DEST, team_abbr="COL",
                      out_png=os.path.join(OUT, "20162017_COL_offence_map.png"),
                      nx=80, ny=54, sigma=1.6, use_contour=True)
    save_team_map_png("20202021", DEST, team_abbr="COL",
                      out_png=os.path.join(OUT, "20202021_COL_offence_map.png"),
                      nx=80, ny=54, sigma=1.6, use_contour=True)
