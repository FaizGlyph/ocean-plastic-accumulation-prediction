
from pathlib import Path

import pandas as pd
import folium
from folium.plugins import HeatMap


# ==================================================
# 1. PATHS
# ==================================================

ROOT = Path(__file__).resolve().parent

PREDICTION_FILE = ROOT / "data" / "prediction_grid.csv"
SUPPORT_FILE = ROOT / "data" / "prediction_grid_support.csv"

OUTPUT_FILE = (
    ROOT / "data" / "ocean_plastic_prediction_map_support.html"
)


# ==================================================
# 2. LOAD SVR PREDICTIONS AND SUPPORT DATA
# ==================================================

predictions = pd.read_csv(PREDICTION_FILE)
support_df = pd.read_csv(SUPPORT_FILE)

print("SVR prediction points:", len(predictions))
print("Support points:", len(support_df))


# ==================================================
# 3. CHECK REQUIRED COLUMNS
# ==================================================

location_columns = [
    "Latitude (degree)",
    "Longitude (degree)",
]

support_columns = [
    "nearest_observation_km",
    "nearest_environment_distance",
    "geographic_support",
    "environment_support",
    "prediction_support",
]

for column in location_columns + ["Predicted_Concentration"]:
    if column not in predictions.columns:
        raise ValueError(
            f"Missing prediction column: {column}"
        )

for column in location_columns + support_columns:
    if column not in support_df.columns:
        raise ValueError(
            f"Missing support column: {column}"
        )


# ==================================================
# 4. MERGE SVR PREDICTIONS WITH SUPPORT DIAGNOSTICS
# ==================================================

# Remove any older prediction values from the support file.
old_prediction_columns = [
    "Prediction",
    "prediction",
    "Predicted Concentration",
    "Predicted_Concentration",
    "predicted_concentration",
    "predicted_plastic",
    "Microplastics Prediction",
    "Accumulation Potential",
    "accumulation_potential",
    "Accumulation_Potential",
]

support_df = support_df.drop(
    columns=[
        col for col in old_prediction_columns
        if col in support_df.columns
    ],
    errors="ignore",
)

df = predictions.merge(
    support_df,
    on=location_columns,
    how="left",
    validate="one_to_one",
)

if len(df) != len(predictions):
    raise ValueError(
        "Merged row count differs from prediction grid. "
        "Check for duplicate coordinates."
    )

if df["prediction_support"].isna().any():
    missing_count = int(df["prediction_support"].isna().sum())
    raise ValueError(
        f"{missing_count} grid points have no support diagnostics. "
        "Regenerate or align the support data before mapping."
    )

print("Merged points:", len(df))


# ==================================================
# 5. ACCUMULATION POTENTIAL FROM SVR
# ==================================================

df["Accumulation_Potential"] = pd.cut(
    df["Predicted_Concentration"],
    bins=[-0.000001, 1, 10, float("inf")],
    labels=["Low", "Medium", "High"],
    right=False,
).astype(str)

# Make sure zero concentration is classified as Low.
df.loc[
    df["Predicted_Concentration"] < 1,
    "Accumulation_Potential"
] = "Low"


# ==================================================
# 6. MAP CENTER
# ==================================================

center_lat = df["Latitude (degree)"].mean()
center_lon = df["Longitude (degree)"].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=5,
    tiles=None,
)


# ==================================================
# 7. BASE MAP
# ==================================================

folium.TileLayer(
    tiles=(
        "https://server.arcgisonline.com/"
        "ArcGIS/rest/services/"
        "World_Street_Map/"
        "MapServer/tile/{z}/{y}/{x}"
    ),
    attr="Esri WorldStreetMap",
    name="Base Map",
    control=True,
).add_to(m)


# ==================================================
# 8. SVR PREDICTION HEATMAP
# ==================================================

prediction_layer = folium.FeatureGroup(
    name="SVR Predicted Concentration",
    show=True,
)

heat_data = df[
    [
        "Latitude (degree)",
        "Longitude (degree)",
        "Predicted_Concentration",
    ]
].values.tolist()

HeatMap(
    heat_data,
    radius=12,
    blur=18,
    min_opacity=0.25,
    max_zoom=8,
).add_to(prediction_layer)

prediction_layer.add_to(m)


# ==================================================
# 9. SUPPORT LAYER
# ==================================================

support_layer = folium.FeatureGroup(
    name="Prediction Support",
    show=False,
)


def support_color(category):
    if category == "Supported":
        return "green"
    elif category == "Caution":
        return "orange"
    elif category == "Extrapolation":
        return "red"
    return "gray"


for _, row in df.iterrows():

    support = row["prediction_support"]
    color = support_color(support)

    prediction = row["Predicted_Concentration"]
    accumulation = row["Accumulation_Potential"]

    popup_html = f"""
    <div style="width:280px;font-family:Arial,sans-serif;">

        <h4>OceanTrace — SVR Prediction</h4>

        <b>Location</b><br>
        Latitude: {row["Latitude (degree)"]:.4f}<br>
        Longitude: {row["Longitude (degree)"]:.4f}

        <hr>

        <b>SVR-predicted concentration</b><br>
        {prediction:.3f} pieces/m³

        <br><br>

        <b>Accumulation potential</b><br>
        {accumulation}

        <hr>

        <b>Prediction support</b><br>
        {support}

        <br><br>

        <b>Nearest plastic observation</b><br>
        {row["nearest_observation_km"]:.1f} km away

        <br><br>

        <b>Environmental similarity distance</b><br>
        {row["nearest_environment_distance"]:.3f}

        <br><br>

        <b>Environmental support</b><br>
        {row["environment_support"]}

        <br><br>

        <b>Geographic support</b><br>
        {row["geographic_support"]}

        <hr>

        <small>
        Concentration is a model prediction, not a direct
        measurement. Support categories are diagnostics,
        not statistical confidence percentages.
        </small>

    </div>
    """

    folium.CircleMarker(
        location=[
            row["Latitude (degree)"],
            row["Longitude (degree)"],
        ],
        radius=2,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.45,
        weight=0.5,
        popup=folium.Popup(
            popup_html,
            max_width=350,
        ),
    ).add_to(support_layer)

support_layer.add_to(m)


# ==================================================
# 10. LEGEND
# ==================================================

legend_html = """
<div style="
position:fixed;
bottom:35px;
left:35px;
width:245px;
background:white;
border:1px solid #999;
border-radius:8px;
z-index:9999;
font-size:13px;
padding:12px;
box-shadow:0 2px 8px rgba(0,0,0,0.2);
">

<b>Prediction Support</b><br><br>

<span style="color:green;">●</span> Supported<br>
<span style="color:orange;">●</span> Caution<br>
<span style="color:red;">●</span> Extrapolation

<hr>

<b>Important</b><br>
Support indicates how well the location and environmental
conditions are represented by the training data. It is not
a statistical confidence percentage.

<br><br>

Concentration values are predicted by SVR and are not
direct observations of plastic.
</div>
"""

m.get_root().html.add_child(
    folium.Element(legend_html)
)


# ==================================================
# 11. LAYER CONTROL
# ==================================================

folium.LayerControl(
    collapsed=False
).add_to(m)


# ==================================================
# 12. SAVE MAP
# ==================================================

m.save(str(OUTPUT_FILE))

print("\n================================")
print("SVR SUPPORT MAP CREATED")
print("================================")

print("Prediction points:", len(df))
print("Prediction range:",
      round(df["Predicted_Concentration"].min(), 3),
      "to",
      round(df["Predicted_Concentration"].max(), 3),
      "pieces/m³")

print("\nSupport distribution:")
print(df["prediction_support"].value_counts())

print("\nSaved:", OUTPUT_FILE)
