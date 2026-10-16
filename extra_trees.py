
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "ml_dataset.csv"
OUTPUT_DIR = ROOT / "model_comparison"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BLOCK_SIZE = 2.0
N_SPLITS = 5

LAT = "Latitude (degree)"
LON = "Longitude (degree)"
TARGET = "Microplastics Measurement"


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Model: Extra Trees Regressor")
print("Dataset shape:", df.shape)
print("Scikit-learn version:", sklearn.__version__)


# --------------------------------------------------
# 3. Define features and target
# --------------------------------------------------

features = [
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

# Keep the original row order and the same complete observations.
df = df.dropna(subset=features + [TARGET]).reset_index(drop=True)

X = df[features].copy()
y = df[TARGET].copy()


# --------------------------------------------------
# 4. Create spatial blocks
# Same method as spatial_cv.py
# --------------------------------------------------

df["lat_block"] = np.floor(
    df[LAT] / BLOCK_SIZE
)

df["lon_block"] = np.floor(
    df[LON] / BLOCK_SIZE
)

df["spatial_block"] = (
    df["lat_block"].astype(str)
    + "_"
    + df["lon_block"].astype(str)
)

groups = df["spatial_block"]

print("Observations:", len(df))
print("Spatial blocks:", groups.nunique())


# --------------------------------------------------
# 5. Spatial 5-fold cross-validation
# --------------------------------------------------

gkf = GroupKFold(n_splits=N_SPLITS)

fold_results = []

for fold, (train_idx, test_idx) in enumerate(
    gkf.split(X, y, groups=groups),
    start=1,
):
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # Extra Trees model
    model = ExtraTreesRegressor(
        n_estimators=200,
        min_samples_leaf=2,
        max_features=1.0,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    fold_results.append({
        "fold": fold,
        "train_samples": len(train_idx),
        "test_samples": len(test_idx),
        "MAE": mae,
        "R2": r2,
    })

    print(f"\nFold {fold}")
    print("Train samples:", len(train_idx))
    print("Test samples :", len(test_idx))
    print("MAE          :", round(mae, 4))
    print("R²           :", round(r2, 4))


# --------------------------------------------------
# 6. Save results
# --------------------------------------------------

results_df = pd.DataFrame(fold_results)

output_file = OUTPUT_DIR / "extra_trees_spatial_cv.csv"
results_df.to_csv(output_file, index=False)


# --------------------------------------------------
# 7. Final summary
# --------------------------------------------------

print("\n==============================")
print("EXTRA TREES SPATIAL CV")
print("==============================")

print(
    "MAE scores:",
    [round(value, 4) for value in results_df["MAE"]]
)

print(
    "Mean MAE:",
    round(results_df["MAE"].mean(), 4)
)

print(
    "R² scores:",
    [round(value, 4) for value in results_df["R2"]]
)

print(
    "Mean R²:",
    round(results_df["R2"].mean(), 4)
)

print("\nResults saved to:", output_file)
