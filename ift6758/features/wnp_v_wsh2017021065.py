# %%
import numpy as np
import pandas as pd
from feature_ingenering import clean_dataframe
import wandb

# %%
# Générer df
df_201618 = clean_dataframe(2016, 2018)

# %%
print(df_201618.shape)

# %%
# Filtrer pour match 2017021065
wpg_v_wsh_2017021065 = df_201618[df_201618["idGame"] == 2017021065].copy()

# %%
print(wpg_v_wsh_2017021065.shape)
print(wpg_v_wsh_2017021065.head())

# %%
# Run WandB
run = wandb.init(project= "IFT6758-2025")

# %%
# Création artéfact WandB
artifact = wandb.Artifact(
    name="wpg_v_wsh_2017021065", 
    type="dataset"
)

# %%
# Créer table WandB et ajouter à l'artefact:
table = wandb.Table(dataframe=wpg_v_wsh_2017021065)
artifact.add(table, "wpg_v_wsh_2017021065")

# %%
# Logger artefact
run.log_artifact(artifact)

# %%
# Log en tant que panel pour visualisation
run.log({"wpg_v_wsh_2017021065": table})

# %%
run.finish()