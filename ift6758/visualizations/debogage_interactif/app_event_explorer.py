import io
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from ift6758.visualizations.debogage_interactif.patinoire import draw_rink
from ift6758.visualizations.debogage_interactif.explorer_core import (
    REGULAR, PLAYOFF, load_game,
    event_xy, event_title, shooter_goalie, event_summary_row, color_for_type,
    filter_events
)

st.set_page_config(page_title="IFT6758 – Explorateur d'événements NHL", layout="wide")

# Sidebar -----------------------------
st.sidebar.header("Paramètres")

def season_list(start=2016, end=2024):
    return [f"{y}{y+1}" for y in range(start, end)]

season = st.sidebar.selectbox("Saison", season_list(), index=season_list().index("20232024"))
season_type_label = st.sidebar.radio("Type", ["Saison régulière", "Séries éliminatoires"], horizontal=False)
season_type = REGULAR if season_type_label.startswith("Saison régulière") else PLAYOFF

max_guess = 1350 if season_type == REGULAR else 500
game_num = st.sidebar.number_input("N° du match (1 → ...)", min_value=1, max_value=max_guess, value=1, step=1)

types_opts = ["shot-on-goal", "goal", "blocked-shot", "missed-shot", "faceoff", "penalty"]
sel_types = st.sidebar.multiselect("Types d'événements", types_opts, default=["shot-on-goal", "goal"])
only_with_xy = st.sidebar.checkbox("Seulement avec coordonnées (x,y)", value=True)

period_opts = ["Tous", 1, 2, 3, 4, 5, 6]  # support prolongations multiples
sel_periods = st.sidebar.multiselect("Périodes", period_opts, default=["Tous"])
period_filter = None if "Tous" in sel_periods else [p for p in sel_periods if isinstance(p, int)]

st.sidebar.caption("Astuce : si le match n'existe pas, l'app affichera simplement 0 évènement.")

# Chargement -------------------------
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

# Navigateur d'événement -------------------------
if "ev_pos" not in st.session_state: st.session_state.ev_pos = 0
st.session_state.ev_pos = min(st.session_state.ev_pos, len(kept_idx)-1)

col_nav = st.columns([1, 6, 1])
with col_nav[0]:
    if st.button("⟨ Précédent", use_container_width=True):
        st.session_state.ev_pos = max(0, st.session_state.ev_pos - 1)
with col_nav[2]:
    if st.button("Suivant ⟩", use_container_width=True):
        st.session_state.ev_pos = min(len(kept_idx)-1, st.session_state.ev_pos + 1)

ev_pos = st.slider("Index (après filtrage)", 0, len(kept_idx)-1, st.session_state.ev_pos, key="slider_evpos")
st.session_state.ev_pos = ev_pos

ev = plays[kept_idx[ev_pos]]
xy = event_xy(ev)
shooter, goalie = shooter_goalie(ev)
title = event_title(ev)

# Tracé patinoire -------------------------
fig, ax = plt.subplots(figsize=(10.5, 4.2))
draw_rink(ax, away_label=meta["away"] or "AWAY", home_label=meta["home"] or "HOME")
if xy:
    color = color_for_type(ev.get("typeDescKey", ""))
    ax.scatter([xy[0]], [xy[1]], s=160, color=color, zorder=5)
st.pyplot(fig, use_container_width=True)

# Télécharger l'image PNG
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
st.download_button("Télécharger la figure (PNG)", data=buf.getvalue(),
                   file_name=f"{meta['game_id']}_{kept_idx[ev_pos]}.png", mime="image/png")

# Résumé & JSON -------------------------
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

# Tableau des évènements filtrés + export -------------------------
rows = [event_summary_row(i, plays[i]) for i in kept_idx]
df = pd.DataFrame(rows)
st.markdown("**Événements filtrés (aperçu)**")
st.dataframe(df.head(25), use_container_width=True)
st.download_button("Exporter évènements filtrés (CSV)", data=df.to_csv(index=False).encode("utf-8"),
                   file_name=f"{meta['game_id']}_filtered_events.csv", mime="text/csv")
