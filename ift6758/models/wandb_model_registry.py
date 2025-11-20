import wandb

ENTITY = "qi-li-1-universit-de-montr-al"
PROJECT = "IFT6758-2025"

#####################################
### REGISTER LOGISTIC REGRESSIONS ###
#####################################

# --- LogReg Distance ---
run = wandb.init(entity=ENTITY, project=PROJECT, name="register-logreg-distance")

artifact = run.use_artifact(f"{ENTITY}/{PROJECT}/logreg-distance:v0", type="model")


# Add alias to mark as "best" or "production"
artifact.aliases.append("best")
artifact.save()
print("Registered Logistic Regression (Distance) model to Model Registry")

run.finish()

# --- LogReg Angle ---
run = wandb.init(entity=ENTITY, project=PROJECT, name="register-logreg-angle")

artifact = run.use_artifact(f"{ENTITY}/{PROJECT}/logreg-angle:v0", type="model")
artifact.aliases.append("best")
artifact.save()
print("Registered Logistic Regression (Angle) model to Model Registry")

run.finish()

# --- LogReg Distance + Angle ---
run = wandb.init(entity=ENTITY, project=PROJECT, name="register-logreg-distance-angle")

artifact = run.use_artifact(f"{ENTITY}/{PROJECT}/logreg-distance-angle:v0", type="model")
artifact.aliases.append("best")
artifact.save()
print("✅ Registered Logistic Regression (Distance + Angle) model to Model Registry")

run.finish()

#####################################
### REGISTER XGBOOST ###
#####################################

run = wandb.init(entity=ENTITY, project=PROJECT, name="register-xgboost")

artifact = run.use_artifact(f"{ENTITY}/{PROJECT}/xgboost_tuned_fs:v0", type="model")
artifact.aliases.append("best")
artifact.save()
print("Registered XGBoost Feature Selection model to Model Registry")

run.finish()

#####################################
### REGISTER ENSEMBLE ###
#####################################

run = wandb.init(entity=ENTITY, project=PROJECT, name="register-ensemble")

artifact = run.use_artifact(f"{ENTITY}/{PROJECT}/ensemble-tuned-model:v0", type="model")
artifact.aliases.append("best")
artifact.save()
print("Registered Ensemble model to Model Registry")

run.finish()