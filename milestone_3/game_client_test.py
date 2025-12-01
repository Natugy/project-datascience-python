#%%
import numpy as np
import pandas as pd
import game_client
from game_client import GameClient
from serving_client import ServingClient   


#%%
# ---------------------------------------------
# 1. Instantiate real ServingClient
# ---------------------------------------------
# Adjust IP/port depending on how your prediction service runs
serving_client = ServingClient(
    ip="0.0.0.0",
    port=5000,
    features=["distance_net", "angle_net", "empty_net"]
)

# ---------------------------------------------
# 2. Create the GameClient
# ---------------------------------------------
game_client = GameClient(serving_client)


# ---------------------------------------------
# 3. Choose a real NHL game ID
# ---------------------------------------------
# Replace with any valid NHL game ID
TEST_GAME_ID = "2024020051"

#%%
# ---------------------------------------------
# 4. STEP-BY-STEP INSPECTION
# ---------------------------------------------
print("\n=== STEP 1: Fetch the game from NHL API ===")
try:
    game_json = game_client.fetch_game(TEST_GAME_ID)
    print("Fetched game successfully.")
    print("Top-level keys:", list(game_json.keys()))
except Exception as e:
    print("Error fetching game:", e)
    exit()

print("\n=== STEP 2: Read extracted metadata ===")
metadata = game_client.get_game_metadata()
print(metadata)

#%%
print("\n=== STEP 3: Extract NEW shot events (first call will show all shots so far) ===")
df_new = game_client.get_new_shots_for_prediction()
print(df_new)

#%%
print("\n=== STEP 4: Now test ping_game() (fetch + extract + predict) ===")
try:
    meta, df_pred = game_client.ping_game(TEST_GAME_ID)

    print("\nReturned metadata:")
    print(meta)

    if df_pred is None:
        print("\nNo new shots found → no predictions.")
    else:
        print("\nPrediction DataFrame from ServingClient:")
        print(df_pred)

except Exception as e:
    print("\nError during prediction:", e)


print("\n=== STEP 5: Call ping_game() again → should return NO new shots ===")
meta2, df_pred2 = game_client.ping_game(TEST_GAME_ID)

print("\nMetadata again:")
print(meta2)

if df_pred2 is None:
    print("\nAs expected: no new shots → no predictions.")
else:
    print("\nUnexpected predictions (means new events occurred):")
    print(df_pred2)
# %%
