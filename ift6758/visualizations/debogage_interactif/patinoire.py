import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc, Wedge

RINK_X = (-100.0, 100.0)    
RINK_Y = (-42.5, 42.5) 

def draw_rink(ax: plt.Axes, away_label=str, home_label=str) -> None:
    # palettes & tailles (teintes proches de l'image de référence)
    RED   = "#e58b8b"   # rouge "rose" des lignes/circles
    BLUE  = "#54a6ff"   # bleu NHL doux
    PALEB = "#bfe3ff"   # remplissage crease
    GREY  = "#c7d2e0"   # montant du but
    CR    = 10        # rayon des coins (ft)
    LW_B  = 2.2         # largeur trait "boards"
    LW_L  = 2.0         # largeur lignes
    LW_C  = 1.5         # largeur cercles

    ax.clear()
    ax.set_aspect("equal")
    ax.set_xlim(*RINK_X)
    ax.set_ylim(*RINK_Y)
    ax.set_facecolor("white") 

    # ---------- fond blanc à coins arrondis ----------
    for cx, cy, a1, a2 in [
        (RINK_X[0]+CR, RINK_Y[0]+CR, 180, 270),
        (RINK_X[0]+CR, RINK_Y[1]-CR,  90, 180),
        (RINK_X[1]-CR, RINK_Y[1]-CR,   0,  90),
        (RINK_X[1]-CR, RINK_Y[0]+CR, 270, 360),
    ]:
        ax.add_patch(Wedge((cx, cy), CR, a1, a2, facecolor="white", edgecolor="none"))

    # contour des bandes
    ax.plot([RINK_X[0]+CR, RINK_X[1]-CR], [RINK_Y[1], RINK_Y[1]], color="black", lw=LW_B)
    ax.plot([RINK_X[0]+CR, RINK_X[1]-CR], [RINK_Y[0], RINK_Y[0]], color="black", lw=LW_B)
    ax.plot([RINK_X[0], RINK_X[0]], [RINK_Y[0]+CR, RINK_Y[1]-CR], color="black", lw=LW_B)
    ax.plot([RINK_X[1], RINK_X[1]], [RINK_Y[0]+CR, RINK_Y[1]-CR], color="black", lw=LW_B)
    for cx, cy, th1, th2 in [
        (RINK_X[0]+CR, RINK_Y[1]-CR,  90, 180),
        (RINK_X[0]+CR, RINK_Y[0]+CR, 180, 270),
        (RINK_X[1]-CR, RINK_Y[1]-CR,   0,  90),
        (RINK_X[1]-CR, RINK_Y[0]+CR, 270, 360),
    ]:
        ax.add_patch(Arc((cx, cy), 2*CR, 2*CR, theta1=th1, theta2=th2, color="black", lw=LW_B))

    # ---------- lignes principales ----------
    # centrale (pointillée fine)
    v = ax.axvline(0, color=RED, linewidth=LW_L)
    v.set_dashes([6, 6])
    # bleues ±25
    for x in (-25, 25):
        ax.axvline(x, color=BLUE, linewidth=LW_L+0.6)
    # lignes de but ±89
    for x in (-89, 89):
        ax.axvline(x, color=RED, linewidth=LW_L)

    # ---------- cercles & points de mise au jeu ----------
    # cercles offensifs/défensifs (centres ~ ±69, ±22 ; R=15)
    o_centers = [( 69,  22), ( 69, -22), (-69,  22), (-69, -22)]
    for cx, cy in o_centers:
        ax.add_patch(Circle((cx, cy), 15, fill=False, linewidth=LW_C, edgecolor=RED))
        ax.add_patch(Circle((cx, cy), 1.0, color=RED))
        # petits "hash marks" internes pour se rapprocher du rendu cible
        for dy in (-1.5, 1.5):
            ax.plot([cx-4.5, cx-1.5], [cy+dy, cy+dy], color=RED, lw=1.0)
            ax.plot([cx+4.5, cx+1.5], [cy+dy, cy+dy], color=RED, lw=1.0)
            ax.plot([cx+dy, cx+dy], [cy-4.5, cy-1.5], color=RED, lw=1.0)
            ax.plot([cx+dy, cx+dy], [cy+4.5, cy+1.5], color=RED, lw=1.0)

    # cercle central + points neutres
    ax.add_patch(Circle((0, 0), 15, fill=False, linewidth=LW_C, edgecolor=BLUE))
    for cx, cy in [(-20, 22), (-20, -22), (20, 22), (20, -22)]:
        ax.add_patch(Circle((cx, cy), 1.0, color=RED))

    # petit demi-cercle "au bas" au centre
    ax.add_patch(Arc((0, RINK_Y[0]), 16, 16, theta1=0, theta2=180, color=RED, lw=1.0))

    # ---------- creases & cages ----------
    # crease (demi-disque bleu clair) + montant du but gris
    ax.add_patch(Wedge(center=(-89, 0), r=6.0, theta1=-90, theta2=90,
                       facecolor=PALEB, edgecolor=BLUE, linewidth=1.2, alpha=0.8))
    ax.add_patch(Wedge(center=( 89, 0), r=6.0, theta1=90,  theta2=-90,
                       facecolor=PALEB, edgecolor=BLUE, linewidth=1.2, alpha=0.8))
    ax.add_patch(Rectangle((-89, 3), -2, -6, fill=False, linewidth=1.4, edgecolor=GREY))
    ax.add_patch(Rectangle(( 89, -3), 2, 6, fill=False, linewidth=1.4, edgecolor=GREY))

    # trapezoid derrière le but 
    # approx: (±89, ±7.5) -> (±100, ±14)
    for s in (-1, 1):
        x0, y0 = -89*s, 7.5*s
        x1, y1 = -100*s, 21.25*(2/3)*s
        ax.plot([x0, x1], [ y0,  y1], color=RED, lw=1.1)
        ax.plot([x0, x1], [-y0, -y1], color=RED, lw=1.1)

    # ---------- libellés & axes ----------
    ax.text(-50, RINK_Y[1]+1.2, away_label, ha="center", va="bottom", fontsize=10)
    ax.text( 50, RINK_Y[1]+1.2, home_label, ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("feet")
    ax.set_ylabel("feet")
    ax.set_xticks(np.linspace(-100, 100, 9))
    ax.set_yticks(np.linspace(-42.5, 42.5, 5))
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(12, 6))
    draw_rink(ax, away_label="WPG", home_label="COL")
    # placer un événement
    ax.scatter([72], [-2], s=120, color="#2155cd")  # ex: un tir côté droit
    plt.show()