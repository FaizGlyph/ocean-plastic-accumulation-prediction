import pandas as pd
import folium
from folium.plugins import HeatMap

# =========================
# LOAD PREDICTION DATA
# =========================

df = pd.read_csv("data/prediction_grid.csv")

# Center of study area
center_lat = df["Latitude (degree)"].mean()
center_lon = df["Longitude (degree)"].mean()

# =========================
# CREATE MAP
# =========================

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=5,
    tiles="Esri.WorldStreetMap"
)

# =========================
# HEATMAP
# =========================

heat_data = [
    [
        row["Latitude (degree)"],
        row["Longitude (degree)"],
        row["Predicted_Concentration"]
    ]
    for _, row in df.iterrows()
]

HeatMap(
    heat_data,
    radius=15,
    blur=20,
    min_opacity=0.3,
    max_zoom=8
).add_to(m)

# =========================
# SAVE MAP
# =========================

m.save(
    "data/ocean_plastic_prediction_map.html"
)

print("MAP CREATED")
print("-----------")
print("Prediction points:", len(df))
print("Saved:")
print("data/ocean_plastic_prediction_map.html")