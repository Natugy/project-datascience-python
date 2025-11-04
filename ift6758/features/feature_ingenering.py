import numpy as np
from ift6758.data.pandas_conversion import *
import matplotlib.pyplot as plt
import seaborn as sns


def clean_dataframe():
    df = get_seasons_dataframe(2016,2020)

    df = df[df["typeDescKey"].isin(["shot-on-goal", "goal"])].copy()
    df["is_goal"] = (df["typeDescKey"] == "goal").astype(int)
    df["empty_net"] = 0
    goal_x = np.where(df["xCoord"] > 0, 89, -89)
    goal_y = 0
    df["distance_net"] = np.sqrt((goal_x - df["xCoord"])**2 + (goal_y - df["yCoord"])**2)
    df["angle_net"] = np.degrees(np.arctan2(np.abs(df["yCoord"]), np.abs(goal_x - df["xCoord"])))
    df_clean = df[["xCoord", "yCoord", "distance_net", "angle_net", "is_goal", "empty_net", "season", "teamAbbr", "idGame"]]

    print(df_clean.head())
    return df_clean
    
def figure_ratio_but_nonbut(df_clean):
    df_clean["distance_net"] = pd.to_numeric(df_clean["distance_net"], errors="coerce")
    df_clean["angle_net"] = pd.to_numeric(df_clean["angle_net"], errors="coerce")

    bin_distance = np.arange(0, 90, 5)
    bin_angle = np.arange(0, 90, 5)

    # === 1️⃣ Taux de but par distance ===
    df_clean["distance_bin"] = pd.cut(df_clean["distance_net"], bins=bin_distance)

    goal_rate_distance = (
        df_clean.groupby("distance_bin")["is_goal"]
        .mean()
        .reset_index()
        .rename(columns={"is_goal": "goal_rate"})
    )

    goal_rate_distance["distance_mid"] = goal_rate_distance["distance_bin"].apply(
        lambda x: x.mid if pd.notnull(x) else np.nan
    )

    plt.figure(figsize=(10,6))
    sns.lineplot(data=goal_rate_distance, x="distance_mid", y="goal_rate", marker="o")
    plt.title("Taux de but en fonction de la distance au filet")
    plt.xlabel("Distance (pieds)")
    plt.ylabel("Taux de but (#buts / #tirs)")
    plt.grid(True)
    plt.show()


    df_clean["angle_bin"] = pd.cut(df_clean["angle_net"], bins=bin_angle)

    goal_rate_angle = (
        df_clean.groupby("angle_bin")["is_goal"]
        .mean()
        .reset_index()
        .rename(columns={"is_goal": "goal_rate"})
    )

    goal_rate_angle["angle_mid"] = goal_rate_angle["angle_bin"].apply(
        lambda x: x.mid if pd.notnull(x) else np.nan
    )

    plt.figure(figsize=(10,6))
    sns.lineplot(data=goal_rate_angle, x="angle_mid", y="goal_rate", marker="o", color="orange")
    plt.title("Taux de but en fonction de l’angle du tir")
    plt.xlabel("Angle (degrés)")
    plt.ylabel("Taux de but (#buts / #tirs)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    clean_dataframe()