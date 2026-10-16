
from pathlib import Path

import pandas as pd
import folium
from folium.plugins import HeatMap


# =========================
# PATHS
# =========================

ROOT = Path(__file__).resolve().parent

INPUT_PATH = ROOT / "data" / "prediction_grid.csv"
OUTPUT_PATH = ROOT / "data" / "ocean_plastic_prediction_map.html"


# =========================
# LOAD SVR PREDICTION DATA
# =========================

df = pd.read_csv(INPUT_PATH)

required_columns = [
    "Latitude (degree)",
    "Longitude (degree)",
    "Predicted_Concentration",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Prediction CSV is missing columns: {missing_columns}"
    )

df = df.dropna(subset=required_columns)

if df.empty:
    raise ValueError("No valid prediction points found.")

# Ensure mapped concentrations are non-negative.
df["Predicted_Concentration"] = (
    df["Predicted_Concentration"].clip(lower=0)
)


# =========================
# MAP CENTER
# =========================

center_lat = df["Latitude (degree)"].mean()
center_lon = df["Longitude (degree)"].mean()


# =========================
# CREATE MAP
# =========================

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=5,
    tiles="Esri.WorldStreetMap",
)



# Title shown on the map
title_html = """
<div style="
    position: fixed;
    top: 12px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 9999;
    background: white;
    padding: 10px 16px;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    font-family: Arial, sans-serif;
">
    <b>OceanTrace</b><br>
    <span style="font-size:12px;">
        SVR-Predicted Microplastic Concentration
    </span>
</div>
"""

m.get_root().html.add_child(folium.Element(title_html))


# =========================
# HEATMAP
# =========================

heat_data = df[
    [
        "Latitude (degree)",
        "Longitude (degree)",
        "Predicted_Concentration",
    ]
].values.tolist()

HeatMap(
    heat_data,
    radius=15,
    blur=20,
    min_opacity=0.3,
    max_zoom=8,
).add_to(m)


# =========================
# SAVE MAP
# =========================

m.save(str(OUTPUT_PATH))

print("SVR MAP CREATED")
print("----------------")
print("Prediction points:", len(df))
print(
    "Concentration range:",
    round(df["Predicted_Concentration"].min(), 3),
    "to",
    round(df["Predicted_Concentration"].max(), 3),
    "pieces/m³",
)
print("Saved:", OUTPUT_PATH)
