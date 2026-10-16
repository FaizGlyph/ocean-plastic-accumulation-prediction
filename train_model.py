
from pathlib import Path

import pandas as pd
import joblib

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "data" / "ml_dataset.csv"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "svr_model.joblib"


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset loaded:", df.shape)


# --------------------------------------------------
# 3. Features and target
# Keep the same feature order as the original model
# --------------------------------------------------

features = [
    "Latitude (degree)",
    "Longitude (degree)",
    "wind_u",
    "wind_v",
    "wind_speed",
    "wave_direction",
    "wave_period",
    "wave_height",
    "current_u",
    "current_v",
    "current_speed",
    "month",
]

target = "Microplastics Measurement"

df = df.dropna(subset=features + [target]).reset_index(drop=True)

X = df[features]
y = df[target]


# --------------------------------------------------
# 4. Build SVR pipeline
# Scaling is essential for SVR
# --------------------------------------------------

model = make_pipeline(
    StandardScaler(),
    SVR(
        kernel="rbf",
        C=10.0,
        epsilon=0.1,
    ),
)


# --------------------------------------------------
# 5. Train final model on all observations
# --------------------------------------------------

model.fit(X, y)


# --------------------------------------------------
# 6. Save model and feature list
# --------------------------------------------------

joblib.dump(
    {
        "model": model,
        "features": features,
        "target": target,
        "model_name": "Support Vector Regression",
    },
    MODEL_PATH,
)


# --------------------------------------------------
# 7. Confirmation
# --------------------------------------------------

print("\nFINAL SVR MODEL TRAINED")
print("-----------------------")
print("Training samples:", len(df))
print("Features:", len(features))
print("Algorithm: Support Vector Regression")
print("Kernel: RBF")
print("Feature scaling: StandardScaler")
print("Saved to:", MODEL_PATH)
print("\nOriginal Random Forest model was not modified.")
