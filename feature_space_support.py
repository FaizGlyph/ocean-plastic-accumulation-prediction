import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


# --------------------------------------------------
# 1. Load datasets
# --------------------------------------------------

grid = pd.read_csv(
    "data/prediction_grid_support.csv"
)

training = pd.read_csv(
    "data/ml_dataset.csv"
)

print("Prediction grid:", len(grid))
print("Training observations:", len(training))


# --------------------------------------------------
# 2. Environmental features ONLY
# --------------------------------------------------

environmental_features = [
    "wind_u",
    "wind_v",
    "wind_speed",
    "wave_direction",
    "wave_period",
    "wave_height",
    "current_u",
    "current_v",
    "current_speed",
    "month"
]


# --------------------------------------------------
# 3. Extract environmental features
# --------------------------------------------------

X_train = training[
    environmental_features
].copy()

X_grid = grid[
    environmental_features
].copy()


# --------------------------------------------------
# 4. Standardize
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_grid_scaled = scaler.transform(
    X_grid
)


# --------------------------------------------------
# 5. Find nearest training condition
# --------------------------------------------------

nn = NearestNeighbors(
    n_neighbors=1,
    metric="euclidean"
)

nn.fit(X_train_scaled)

distances, indices = nn.kneighbors(
    X_grid_scaled
)

feature_distance = distances[:, 0]


# --------------------------------------------------
# 6. Add environmental support information
# --------------------------------------------------

grid["nearest_environment_distance"] = (
    feature_distance
)

grid["nearest_environment_index"] = (
    indices[:, 0]
)


# --------------------------------------------------
# 7. Save
# --------------------------------------------------

output_file = (
    "data/prediction_grid_support.csv"
)

grid.to_csv(
    output_file,
    index=False
)


# --------------------------------------------------
# 8. Diagnostics
# --------------------------------------------------

print("\n================================")
print("ENVIRONMENTAL SUPPORT DIAGNOSTIC")
print("================================")

print("\nEnvironmental feature distance:")

print(
    "Minimum:",
    round(feature_distance.min(), 4)
)

print(
    "Maximum:",
    round(feature_distance.max(), 4)
)

print(
    "Mean:",
    round(feature_distance.mean(), 4)
)

print(
    "Median:",
    round(np.median(feature_distance), 4)
)


# --------------------------------------------------
# 9. Percentiles
# --------------------------------------------------

percentiles = [
    10, 25, 50, 75, 90, 95, 99
]

print("\nDistance percentiles:")

for p in percentiles:

    value = np.percentile(
        feature_distance,
        p
    )

    print(
        f"{p}th percentile:",
        round(value, 4)
    )


print("\nSaved:")
print(output_file)