import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("data/ml_dataset.csv")

print("Dataset shape:", df.shape)


# --------------------------------------------------
# 2. Define features and target
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


X = df[features].copy()
y = df[target].copy()


# --------------------------------------------------
# 3. Create spatial blocks
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


print("\nNumber of spatial blocks:", groups.nunique())


# --------------------------------------------------
# 4. Spatial 5-fold CV
# --------------------------------------------------

gkf = GroupKFold(n_splits=5)

mae_scores = []
r2_scores = []


for fold, (train_idx, test_idx) in enumerate(
    gkf.split(X, y, groups=groups),
    start=1
):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]


    # --------------------------------------------------
    # 5. Train Random Forest
    # --------------------------------------------------

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)


    # --------------------------------------------------
    # 6. Predict validation region
    # --------------------------------------------------

    y_pred = model.predict(X_test)


    # --------------------------------------------------
    # 7. Evaluate
    # --------------------------------------------------

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mae_scores.append(mae)
    r2_scores.append(r2)


    print(f"\nFold {fold}")
    print("Train samples:", len(train_idx))
    print("Test samples :", len(test_idx))
    print("MAE          :", round(mae, 4))
    print("R²           :", round(r2, 4))


# --------------------------------------------------
# 8. Final results
# --------------------------------------------------

print("\n==============================")
print("SPATIAL CROSS-VALIDATION")
print("==============================")

print(
    "MAE scores:",
    [round(x, 4) for x in mae_scores]
)

print(
    "Mean MAE:",
    round(np.mean(mae_scores), 4)
)

print(
    "R² scores:",
    [round(x, 4) for x in r2_scores]
)

print(
    "Mean R²:",
    round(np.mean(r2_scores), 4)
)