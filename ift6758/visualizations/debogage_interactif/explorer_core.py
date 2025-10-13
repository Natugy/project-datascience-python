from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from ift6758.data.data_scrapping import LNHDataScrapper

# Types de saison NHL
REGULAR = "02"
PLAYOFF = "03"

# ------------------------- bornes régulière par saison -------------------------

def regular_max_guess(season: str) -> int:
    """Borne supérieure plausible du nombre de matchs de saison régulière pour une saison donnée."""
    first = int(season[:4])
    # approx nb équipes
    if first >= 2021: teams = 32
    elif first >= 2017: teams = 31
    else: teams = 30
    special = {}
    return special.get(season, (teams * 82) // 2)

# ------------------------- utilitaires gameId -------------------------

def build_game_id(season: str, season_type: str, game_number: int) -> str:
    """
    season      : '20172018'
    season_type : '02' (REGULAR) | '03' (PLAYOFF)
    game_number : 1..N (REGULAR) ou 0RMG pour PLAYOFF (R=1..4, M selon ronde, G=1..7)
    """
    return f"{season[:4]}{season_type}{int(game_number):04d}"

def playoff_number(round_: int, matchup: int, game_in_series: int) -> int:
    """Construit 0RMG -> ex: finale G7 = 0417 (R=4,M=1,G=7)."""
    return int(f"0{round_}{matchup}{game_in_series}")

# ------------------------- chargement données -------------------------

def _try_load_local_json(game_id: str, dest_folder: str) -> Optional[Dict[str, Any]]:
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
    """Retourne le JSON play-by-play (cache mémoire + disque ./ressources/)."""
    scr = LNHDataScrapper()
    data = _try_load_local_json(game_id, scr.dest_folder)
    if data is None:
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
    """Retourne (meta, plays) pour l'UI."""
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

# ------------------------- lecture d’un événement -------------------------

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
    s = (ev.get("details") or {}).get("strength")
    return s.upper() if isinstance(s, str) else None

def event_zone(ev: Dict[str, Any]) -> Optional[str]:
    z = (ev.get("details") or {}).get("zoneCode")
    return z if isinstance(z, str) else None

def event_shot_type(ev: Dict[str, Any]) -> Optional[str]:
    s = (ev.get("details") or {}).get("shotType")
    return s if isinstance(s, str) else None

def event_empty_net(ev: Dict[str, Any]) -> Optional[bool]:
    en = (ev.get("details") or {}).get("emptyNet")
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

# ------------------------- filtrage -------------------------

def filter_events(
    plays: List[Dict[str, Any]],
    only_with_xy: bool,
    types: Optional[List[str]] = None,
    period_filter: Optional[List[int]] = None,
) -> List[int]:
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

# palette couleur par type (UI)
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
