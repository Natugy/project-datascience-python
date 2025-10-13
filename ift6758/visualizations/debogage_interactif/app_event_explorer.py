'''
User guide to run this Streamlit app (from the root of the repo):
(ift6758-venv) PS D:\Bureau\project-datascience-python> $env:PYTHONPATH="$PWD"
(ift6758-venv) PS D:\Bureau\project-datascience-python> streamlit run ift6758\visualizations\debogage_interactif\app_event_explorer.py
'''

import os 
import pandas as pd
import matplotlib.pyplot as plt
import io
import streamlit as st

from ift6758.visualizations.debogage_interactif.patinoire import draw_rink
from ift6758.visualizations.debogage_interactif.explorer_core import (
    REGULAR, PLAYOFF, load_game, regular_max_guess,
    playoff_number, color_for_type,
    event_xy, event_title, shooter_goalie, event_summary_row, filter_events
)
from ift6758.data.data_scrapping import LNHDataScrapper

st.set_page_config(page_title="IFT6758 – Explorateur d'événements NHL", layout="wide")

# ------------------------- sidebar -------------------------
st.sidebar.header("Paramètres")

def season_list(start=2016, end=2024):
    return [f"{y}{y+1}" for y in range(start, end)]

season = st.sidebar.selectbox("Saison", season_list(), index=season_list().index("20232024"))
season_type_label = st.sidebar.radio("Type", ["Saison régulière", "Séries éliminatoires"], horizontal=False)
season_type = REGULAR if season_type_label.startswith("Saison régulière") else PLAYOFF

if season_type == REGULAR:
    max_guess = regular_max_guess(season)
    game_num = st.sidebar.number_input("Match # (1 → max)", min_value=1, max_value=max_guess, value=1, step=1)
else:
    # sélecteurs Ronde/Duel/Match pour les playoffs
    round_ = st.sidebar.selectbox("Ronde", [1, 2, 3, 4], index=0)
    matchups_by_round = {1: list(range(1, 9)), 2: list(range(1, 5)), 3: [1, 2], 4: [1]}
    matchup = st.sidebar.selectbox("Duel", matchups_by_round[round_], index=0)
    game_in_series = st.sidebar.slider("Match (1–7)", 1, 7, 1)
    game_num = playoff_number(round_, matchup, game_in_series)

types_opts = ["shot-on-goal", "goal", "blocked-shot", "missed-shot", "faceoff", "penalty"]
sel_types = st.sidebar.multiselect("Types d'événements", types_opts, default=["shot-on-goal", "goal"])
only_with_xy = st.sidebar.checkbox("Seulement avec coordonnées (x,y)", value=True)
period_opts = ["Tous", 1, 2, 3, 4, 5, 6]
sel_periods = st.sidebar.multiselect("Périodes", period_opts, default=["Tous"])
period_filter = None if "Tous" in sel_periods else [p for p in sel_periods if isinstance(p, int)]

# ------------------------- onglets -------------------------
tab_evt, tab_season = st.tabs(["🧭 Événements (débogage)", "📦 Vue saison (CSV)"])

# ========== Onglet 1 : exploration événement ==========
with tab_evt:
    meta, plays = load_game(season, season_type, int(game_num))
    st.title("Explorateur d'événements NHL – IFT6758")
    c1, c2, c3, c4, c5 = st.columns([1.3, 1, 1, 1, 1.2])
    with c1: st.markdown(f"**Game ID** : `{meta['game_id']}`")
    with c2: st.markdown(f"**Date** : {meta['date'] or '—'}")
    with c3: st.markdown(f"**Home** : {meta['home'] or '—'}")
    with c4: st.markdown(f"**Away** : {meta['away'] or '—'}")
    with c5: st.markdown(f"**#Événements** : {meta['n_events']}")

    kept_idx = filter_events(plays, only_with_xy=only_with_xy, types=sel_types or None, period_filter=period_filter)
    if not kept_idx:
        st.warning("Aucun événement à afficher avec ces filtres.")
        st.stop()

    # Navigation
    if "ev_pos" not in st.session_state: st.session_state.ev_pos = 0
    st.session_state.ev_pos = min(st.session_state.ev_pos, len(kept_idx)-1)

    navL, navC, navR = st.columns([1, 6, 1])
    with navL:
        if st.button("⟨ Précédent", use_container_width=True):
            st.session_state.ev_pos = max(0, st.session_state.ev_pos - 1)
    with navR:
        if st.button("Suivant ⟩", use_container_width=True):
            st.session_state.ev_pos = min(len(kept_idx)-1, st.session_state.ev_pos + 1)

    ev_pos = st.slider("Index (après filtre)", 0, len(kept_idx)-1, st.session_state.ev_pos, key="slider_evpos")
    st.session_state.ev_pos = ev_pos

    ev = plays[kept_idx[ev_pos]]
    xy = event_xy(ev)
    shooter, goalie = shooter_goalie(ev)
    title = event_title(ev)

    # Rink + point
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    draw_rink(ax, away_label=meta["away"] or "AWAY", home_label=meta["home"] or "HOME")
    if xy:
        color = color_for_type(ev.get("typeDescKey", ""))
        ax.scatter([xy[0]], [xy[1]], s=160, color=color, zorder=5)
    st.pyplot(fig, use_container_width=True)

    # PNG
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    st.download_button("Télécharger la figure (PNG)", data=buf.getvalue(),
                       file_name=f"{meta['game_id']}_{kept_idx[ev_pos]}.png", mime="image/png")

    # Résumé + JSON
    st.subheader("Résumé de l'événement sélectionné")
    left, right = st.columns([1.3, 1])
    with left:
        st.write(f"**{title}**")
        st.write(f"**Type** : `{ev.get('typeDescKey')}`")
        if xy: st.write(f"**Coordonnées** : x={xy[0]:.1f}, y={xy[1]:.1f}")
        if shooter: st.write(f"**Shooter/Scorer** : {shooter}")
        if goalie:  st.write(f"**Goalie** : {goalie}")
    with right:
        st.json(ev)

    # Tableau + export CSV des événements filtrés
    rows = [event_summary_row(i, plays[i]) for i in kept_idx]
    df = pd.DataFrame(rows)
    st.markdown("**Événements filtrés (aperçu)**")
    st.dataframe(df.head(25), use_container_width=True)
    st.download_button("Exporter événements filtrés (CSV)",
                       data=df.to_csv(index=False).encode("utf-8"),
                       file_name=f"{meta['game_id']}_filtered_events.csv",
                       mime="text/csv")

# ========== Onglet 2 : vue saison (rapide depuis ton CSV) ==========
with tab_season:
    scr = LNHDataScrapper()
    csv_path = os.path.join(scr.dest_folder, f"{season}.csv")
    if not os.path.exists(csv_path):
        st.info("CSV saison introuvable dans ./ressources/. Lance d'abord ton pipeline CSV pour cette saison.")
    else:
        df_season = pd.read_csv(csv_path)
        # filtres simples
        c1, c2 = st.columns(2)
        with c1:
            team_ids = sorted([x for x in df_season["teamId"].dropna().unique().tolist() if str(x) != "nan"])
            team_sel = st.multiselect("Filtrer teamId", team_ids, default=team_ids[:1] if team_ids else [])
        with c2:
            type_sel = st.multiselect("Types (CSV)", sorted(df_season["typeDescKey"].dropna().unique().tolist()),
                                      default=["shot-on-goal", "goal"])
        q = df_season.copy()
        if team_sel: q = q[q["teamId"].isin(team_sel)]
        if type_sel: q = q[q["typeDescKey"].isin(type_sel)]
        st.write(f"**Lignes** : {len(q)}")
        st.dataframe(q.head(50), use_container_width=True)
        st.download_button("Exporter (CSV filtré saison)",
                           data=q.to_csv(index=False).encode("utf-8"),
                           file_name=f"{season}_filtered.csv",
                           mime="text/csv")
