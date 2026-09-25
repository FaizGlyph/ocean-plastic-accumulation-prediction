import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/ml_dataset.csv")

# Same block size used in spatial CV
BLOCK_SIZE = 2.0

# Create spatial blocks
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

# Plot
plt.figure(figsize=(12, 8))

scatter = plt.scatter(
    df["Longitude (degree)"],
    df["Latitude (degree)"],
    c=pd.factorize(df["spatial_block"])[0],
    s=30,
    alpha=0.8
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Spatial Distribution of Plastic Observations and 2° Blocks")

plt.grid(True, alpha=0.3)

plt.tight_layout()

# Save image
plt.savefig(
    "spatial_blocks.png",
    dpi=200
)

plt.show()

print("================================")
print("SPATIAL BLOCK DIAGNOSTIC")
print("================================")

print("Observations:", len(df))
print("Spatial blocks:", df["spatial_block"].nunique())

print("\nObservations per block:")
print(
    df["spatial_block"]
    .value_counts()
    .describe()
)

print("\nSmallest blocks:")
print(
    df["spatial_block"]
    .value_counts()
    .sort_values()
    .head(10)
)

print("\nLargest blocks:")
print(
    df["spatial_block"]
    .value_counts()
    .sort_values(ascending=False)
    .head(10)
)

print("\nSaved:")
print("spatial_blocks.png")