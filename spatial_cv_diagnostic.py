import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("data/ml_dataset.csv")


# --------------------------------------------------
# 2. Features and target
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
    "Latitude (degree)",
    "Longitude (degree)"
]

target = "Microplastics Measurement"

X = df[features]
y = df[target]


# --------------------------------------------------
# 3. Create 2° spatial blocks
# --------------------------------------------------

BLOCK_SIZE = 2.0

df["lat_block"] = np.floor(
    df["Latitude (degree)"] / BLOCK_SIZE
)

df["lon_block"] = np.floor(
    df["Longitude (degree)"] / BLOCK_SIZE
)

df["spatial_block"] = (
    df["lat_block"].astype(str)
    + "_"
    + df["lon_block"].astype(str)
)

groups = df["spatial_block"]


# --------------------------------------------------
# 4. Spatial CV
# --------------------------------------------------

gkf = GroupKFold(n_splits=5)

all_results = []


for fold, (train_idx, test_idx) in enumerate(
    gkf.split(X, y, groups=groups),
    start=1
):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]


    # Train
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)


    # Predict
    y_pred = model.predict(X_test)


    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)


    # Baseline: predict the TRAINING mean
    baseline_pred = np.full(
        len(y_test),
        y_train.mean()
    )

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_pred
    )

    baseline_r2 = r2_score(
        y_test,
        baseline_pred
    )


    print("\n================================")
    print(f"FOLD {fold}")
    print("================================")

    print("Train samples:", len(train_idx))
    print("Test samples :", len(test_idx))

    print("\nTarget distribution")
    print(
        "Train mean:",
        round(y_train.mean(), 3)
    )

    print(
        "Test mean :",
        round(y_test.mean(), 3)
    )

    print(
        "Test std  :",
        round(y_test.std(), 3)
    )

    print(
        "Test min  :",
        round(y_test.min(), 3)
    )

    print(
        "Test max  :",
        round(y_test.max(), 3)
    )

    print("\nRandom Forest")
    print(
        "MAE:",
        round(mae, 4)
    )

    print(
        "R² :",
        round(r2, 4)
    )

    print("\nMean baseline")
    print(
        "MAE:",
        round(baseline_mae, 4)
    )

    print(
        "R² :",
        round(baseline_r2, 4)
    )


    all_results.append({
        "fold": fold,
        "train_mean": y_train.mean(),
        "test_mean": y_test.mean(),
        "test_std": y_test.std(),
        "test_min": y_test.min(),
        "test_max": y_test.max(),
        "model_mae": mae,
        "model_r2": r2,
        "baseline_mae": baseline_mae,
        "baseline_r2": baseline_r2
    })


# --------------------------------------------------
# 5. Summary
# --------------------------------------------------

results = pd.DataFrame(all_results)

print("\n\n================================")
print("SUMMARY")
print("================================")

print(results.round(4).to_string(index=False))

print("\nMean model MAE:",
      round(results["model_mae"].mean(), 4))

print("Mean model R²:",
      round(results["model_r2"].mean(), 4))

print("\nMean baseline MAE:",
      round(results["baseline_mae"].mean(), 4))

print("Mean baseline R²:",
      round(results["baseline_r2"].mean(), 4))