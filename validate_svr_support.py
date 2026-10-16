
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score


ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "data" / "ml_dataset.csv"
SUPPORT_PATH = ROOT / "data" / "prediction_grid_support.csv"
OUTPUT_DIR = ROOT / "model_comparison"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BLOCK_SIZE = 2.0
N_SPLITS = 5

LAT = "Latitude (degree)"
LON = "Longitude (degree)"
TARGET = "Microplastics Measurement"

FEATURES = [
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
    LAT,
    LON,
]


# 1. Load observations and support diagnostics
df = pd.read_csv(DATA_PATH)
support = pd.read_csv(SUPPORT_PATH)

required_support = [
    "Latitude (degree)",
    "Longitude (degree)",
    "prediction_support",
]

missing = [c for c in required_support if c not in support.columns]
if missing:
    raise ValueError(f"Support CSV is missing columns: {missing}")

df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)

X = df[FEATURES].copy()
y = df[TARGET].copy()

# 2. Recreate the spatial folds used by svr.py
df["lat_block"] = np.floor(df[LAT] / BLOCK_SIZE)
df["lon_block"] = np.floor(df[LON] / BLOCK_SIZE)
groups = (
    df["lat_block"].astype(str)
    + "_"
    + df["lon_block"].astype(str)
)

gkf = GroupKFold(n_splits=N_SPLITS)

# 3. Generate out-of-fold predictions for every observation
oof_pred = np.full(len(df), np.nan)
fold_ids = np.full(len(df), -1)

for fold, (train_idx, test_idx) in enumerate(
    gkf.split(X, y, groups=groups), start=1
):
    model = make_pipeline(
        StandardScaler(),
        SVR(kernel="rbf", C=10.0, epsilon=0.1),
    )

    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    oof_pred[test_idx] = model.predict(X.iloc[test_idx])
    fold_ids[test_idx] = fold

df["SVR_OOF_Prediction"] = oof_pred
df["Absolute_Error"] = np.abs(y.to_numpy() - oof_pred)
df["Spatial_Fold"] = fold_ids

# 4. Assign each real observation the support category of its
# nearest prediction-grid point, using geographic distance.
from sklearn.neighbors import BallTree

grid_coords = np.radians(
    support[[LAT, LON]].to_numpy()
)
obs_coords = np.radians(
    df[[LAT, LON]].to_numpy()
)

tree = BallTree(grid_coords, metric="haversine")
distances, indices = tree.query(obs_coords, k=1)

df["Nearest_Grid_Distance_km"] = distances[:, 0] * 6371.0
df["prediction_support"] = (
    support.iloc[indices[:, 0]]["prediction_support"].to_numpy()
)

# 5. Save individual predictions and errors
predictions_file = OUTPUT_DIR / "svr_support_oof_predictions.csv"
df.to_csv(predictions_file, index=False)

# 6. Summarize errors by assigned support category
summary = (
    df.groupby("prediction_support", dropna=False)
    .agg(
        Samples=("Absolute_Error", "size"),
        MAE=("Absolute_Error", "mean"),
        Median_Absolute_Error=("Absolute_Error", "median"),
        Mean_Actual=(TARGET, "mean"),
        Mean_Prediction=("SVR_OOF_Prediction", "mean"),
    )
    .reset_index()
)

summary_file = OUTPUT_DIR / "svr_support_error_validation.csv"
summary.to_csv(summary_file, index=False)

print("\n======================================")
print("SVR SUPPORT-ERROR VALIDATION")
print("======================================")
print("Observations:", len(df))
print("Spatial folds:", N_SPLITS)
print("\nOverall out-of-fold MAE:",
      round(mean_absolute_error(y, oof_pred), 4))
print("Overall out-of-fold R²:",
      round(r2_score(y, oof_pred), 4))

print("\nError by assigned support category:")
print(summary.to_string(
    index=False,
    formatters={
        "MAE": "{:.4f}".format,
        "Median_Absolute_Error": "{:.4f}".format,
        "Mean_Actual": "{:.4f}".format,
        "Mean_Prediction": "{:.4f}".format,
    },
))

print("\nSaved predictions:", predictions_file)
print("Saved summary:", summary_file)

print(
    "\nNote: support labels are assigned from the nearest grid point. "
    "This is an exploratory diagnostic, not calibrated prediction confidence."
)
