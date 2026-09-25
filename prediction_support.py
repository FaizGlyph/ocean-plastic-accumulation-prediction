import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree


# --------------------------------------------------
# 1. Load files
# --------------------------------------------------

grid = pd.read_csv("data/prediction_grid.csv")
plastic = pd.read_csv("data/ml_dataset.csv")

print("Prediction grid:", len(grid))
print("Plastic observations:", len(plastic))


# --------------------------------------------------
# 2. Get coordinates
# --------------------------------------------------

grid_coords = grid[
    ["Latitude (degree)", "Longitude (degree)"]
].values

plastic_coords = plastic[
    ["Latitude (degree)", "Longitude (degree)"]
].values


# --------------------------------------------------
# 3. Convert coordinates to radians
# --------------------------------------------------

grid_rad = np.radians(grid_coords)
plastic_rad = np.radians(plastic_coords)


# --------------------------------------------------
# 4. Find nearest real plastic observation
# --------------------------------------------------

tree = BallTree(
    plastic_rad,
    metric="haversine"
)

distances, indices = tree.query(
    grid_rad,
    k=1
)


EARTH_RADIUS_KM = 6371.0

nearest_distance_km = (
    distances[:, 0] * EARTH_RADIUS_KM
)

nearest_index = indices[:, 0]


# --------------------------------------------------
# 5. Add geographic support information
# --------------------------------------------------

grid["nearest_observation_km"] = (
    nearest_distance_km
)

grid["nearest_observation_index"] = (
    nearest_index
)


# --------------------------------------------------
# 6. Assign geographic support category
# --------------------------------------------------

def classify_support(distance):

    if distance <= 100:
        return "Supported"

    elif distance <= 200:
        return "Caution"

    else:
        return "Extrapolation"


grid["geographic_support"] = (
    grid["nearest_observation_km"]
    .apply(classify_support)
)


# --------------------------------------------------
# 7. Save updated grid
# --------------------------------------------------

output_file = (
    "data/prediction_grid_support.csv"
)

grid.to_csv(
    output_file,
    index=False
)


# --------------------------------------------------
# 8. Print summary
# --------------------------------------------------

print("\n================================")
print("GEOGRAPHIC SUPPORT")
print("================================")

print("\nSupport categories:")

counts = (
    grid["geographic_support"]
    .value_counts()
)

percentages = (
    grid["geographic_support"]
    .value_counts(normalize=True) * 100
)


for category in [
    "Supported",
    "Caution",
    "Extrapolation"
]:

    print(
        f"{category:15}: "
        f"{counts.get(category, 0):5} points "
        f"({percentages.get(category, 0):.1f}%)"
    )


print("\nDistance statistics:")

print(
    "Minimum:",
    round(nearest_distance_km.min(), 2),
    "km"
)

print(
    "Median:",
    round(np.median(nearest_distance_km), 2),
    "km"
)

print(
    "Maximum:",
    round(nearest_distance_km.max(), 2),
    "km"
)


print("\nSaved:")
print(output_file)