from __future__ import annotations
import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from ift6758.data.data_scrapping import LNHDataScrapper

# Types de saison NHL (API gameId = {YYYY}{02/03}{####})
REGULAR = "02"
PLAYOFF = "03"

# ------------------------- Utilitaires gameId -------------------------

def build_game_id(season: str, season_type: str, game_number: int) -> str:
    """
    season       : '20172018'
    season_type  : '02' (REGULAR) | '03' (PLAYOFF)
    game_number  : 1..N
    """
    return f"{season[:4]}{season_type}{int(game_number):04d}"

# ------------------------- Chargement des données -------------------------

def _try_load_local(game_id: str, dest_folder: str) -> Optional[Dict[str, Any]]:
    """Charge ./ressources/{game_id}.json s'il existe (évite les appels réseau)."""
    path = os.path.join(dest_folder, f"{game_id}.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None

@lru_cache(maxsize=4096)
def _fetch_game_cached(game_id: str) -> Dict[str, Any]:
    """
    Retourne le JSON play-by-play pour un game_id NHL.
    Priorité au cache disque local (./ressources/) créé par ton scraper.
    """
    scr = LNHDataScrapper()  # définit dest_folder = ./ressources/
    data = _try_load_local(game_id, scr.dest_folder)
    if data is None:
        # On récupère sur l’API et on sauvegarde localement pour les prochaines fois
        data = scr.get_one_game(game_id, save=True) or {}
    if not isinstance(data.get("plays"), list):
        data["plays"] = []
    return data

def _team_name(team_dict: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(team_dict, dict):
        return None
    return (
        team_dict.get("name")
        or (team_dict.get("placeName") or {}).get("default")
        or team_dict.get("abbrev")
    )

def load_game(season: str, season_type: str, game_number: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Retourne (meta, plays) pour l'UI Streamlit."""
    gid = build_game_id(season, season_type, game_number)
    data = _fetch_game_cached(gid)
    plays = data.get("plays", [])

    meta = {
        "game_id": gid,
        "date": data.get("gameDate") or data.get("gameDateUTC"),
        "home": _team_name(data.get("homeTeam") or data.get("home")),
        "away": _team_name(data.get("awayTeam") or data.get("away")),
        "venue": (data.get("venue") or {}).get("default") if isinstance(data.get("venue"), dict) else data.get("venue"),
        "n_events": len(plays),
    }
    return meta, plays

# ------------------------- Lecture d’un événement -------------------------

def event_xy(ev: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    det = ev.get("details") or {}
    x, y = det.get("xCoord"), det.get("yCoord")
    if x is None or y is None:
        return None
    try:
        return float(x), float(y)
    except Exception:
        return None

def event_period(ev: Dict[str, Any]) -> Optional[int]:
    about = ev.get("about") or {}
    p = about.get("period")
    try:
        return int(p)
    except Exception:
        return None

def event_clock(ev: Dict[str, Any]) -> str:
    about = ev.get("about") or {}
    return about.get("periodTime") or about.get("time") or ""

def event_type(ev: Dict[str, Any]) -> str:
    return ev.get("typeDescKey") or "event"

def event_title(ev: Dict[str, Any]) -> str:
    t = event_type(ev)
    p = event_period(ev)
    return f"{t} — {event_clock(ev)}{(' P'+str(p)) if p else ''}"

def shooter_goalie(ev: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    shooter = None
    goalie = None
    for pl in ev.get("players") or []:
        role = (pl.get("playerType") or "").lower()
        name = (pl.get("player") or {}).get("fullName")
        if role in {"scorer", "shooter"}:
            shooter = name
        elif role in {"goalie", "goaltender"}:
            goalie = name
    return shooter, goalie

def event_strength(ev: Dict[str, Any]) -> Optional[str]:
    # souvent présent pour les buts
    det = ev.get("details") or {}
    s = det.get("strength")
    return s.upper() if isinstance(s, str) else None

def event_zone(ev: Dict[str, Any]) -> Optional[str]:
    det = ev.get("details") or {}
    z = det.get("zoneCode")
    return z if isinstance(z, str) else None

def event_shot_type(ev: Dict[str, Any]) -> Optional[str]:
    det = ev.get("details") or {}
    s = det.get("shotType")
    return s if isinstance(s, str) else None

def event_empty_net(ev: Dict[str, Any]) -> Optional[bool]:
    det = ev.get("details") or {}
    en = det.get("emptyNet")
    return bool(en) if en is not None else None

def event_summary_row(idx: int, ev: Dict[str, Any]) -> Dict[str, Any]:
    x, y = event_xy(ev) or (None, None)
    shooter, goalie = shooter_goalie(ev)
    return dict(
        idx=idx,
        period=event_period(ev),
        time=event_clock(ev),
        type=event_type(ev),
        x=x, y=y,
        zone=event_zone(ev),
        shotType=event_shot_type(ev),
        strength=event_strength(ev),
        emptyNet=event_empty_net(ev),
        shooter=shooter,
        goalie=goalie,
    )

# ------------------------- Filtrage -------------------------

def filter_events(
    plays: List[Dict[str, Any]],
    only_with_xy: bool,
    types: Optional[List[str]] = None,
    period_filter: Optional[List[int]] = None,
) -> List[int]:
    """Renvoie les indices conservés après filtres."""
    keep: List[int] = []
    for i, ev in enumerate(plays):
        if types and (event_type(ev) not in types):
            continue
        if period_filter and (event_period(ev) not in period_filter):
            continue
        if only_with_xy and event_xy(ev) is None:
            continue
        keep.append(i)
    return keep

# Palette couleur par type (cohérente UI)
EVENT_COLOR = {
    "goal": "#1e88e5",
    "shot-on-goal": "#1e88e5",
    "blocked-shot": "#8e24aa",
    "missed-shot": "#fb8c00",
    "faceoff": "#6d4c41",
    "penalty": "#e53935",
}
def color_for_type(t: str) -> str:
    return EVENT_COLOR.get(t, "#1e88e5")
