import pandas as pd
import folium
from folium.plugins import HeatMap


# ==================================================
# 1. LOAD DATA
# ==================================================

file_path = "data/prediction_grid_support.csv"

df = pd.read_csv(file_path)

print("Prediction points:", len(df))

print("\nColumns available:")
print(df.columns.tolist())


# ==================================================
# 2. CHECK REQUIRED COLUMNS
# ==================================================

required_columns = [
    "Latitude (degree)",
    "Longitude (degree)",
    "nearest_observation_km",
    "nearest_environment_distance",
    "geographic_support",
    "environment_support",
    "prediction_support"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing required column: {column}"
        )


# ==================================================
# 3. FIND PREDICTION COLUMN
# ==================================================

possible_prediction_columns = [
    "Prediction",
    "prediction",
    "Predicted Concentration",
    "Predicted_Concentration",
    "predicted_concentration",
    "predicted_plastic",
    "Microplastics Prediction"
]

prediction_column = None

for column in possible_prediction_columns:

    if column in df.columns:

        prediction_column = column
        break


if prediction_column is None:

    raise ValueError(
        "Could not find the prediction column. "
        "Check the columns printed above."
    )


print(
    "\nUsing prediction column:",
    prediction_column
)


# ==================================================
# 4. FIND ACCUMULATION CATEGORY
# ==================================================

possible_category_columns = [
    "Accumulation Potential",
    "accumulation_potential",
    "Accumulation_Potential"
]

category_column = None

for column in possible_category_columns:

    if column in df.columns:

        category_column = column
        break


if category_column is None:

    print(
        "\nWARNING: Accumulation category column "
        "was not found."
    )


# ==================================================
# 5. MAP CENTER
# ==================================================

center_lat = df[
    "Latitude (degree)"
].mean()

center_lon = df[
    "Longitude (degree)"
].mean()


m = folium.Map(
    location=[
        center_lat,
        center_lon
    ],
    zoom_start=5,
    tiles=None
)


# ==================================================
# 6. BASE MAP
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
    control=True
).add_to(m)


# ==================================================
# 7. PREDICTION HEATMAP
# ==================================================

prediction_layer = folium.FeatureGroup(
    name="Predicted Concentration",
    show=True
)


heat_data = []

for _, row in df.iterrows():

    heat_data.append([
        row["Latitude (degree)"],
        row["Longitude (degree)"],
        row[prediction_column]
    ])


HeatMap(
    heat_data,
    radius=12,
    blur=18,
    min_opacity=0.25,
    max_zoom=8
).add_to(prediction_layer)


prediction_layer.add_to(m)


# ==================================================
# 8. SUPPORT LAYER
# ==================================================

support_layer = folium.FeatureGroup(
    name="Prediction Support",
    show=False
)


def support_color(category):

    if category == "Supported":
        return "green"

    elif category == "Caution":
        return "orange"

    elif category == "Extrapolation":
        return "red"

    return "gray"


# ==================================================
# 9. ADD SUPPORT POINTS
# ==================================================

for _, row in df.iterrows():

    support = row[
        "prediction_support"
    ]

    color = support_color(
        support
    )


    # ----------------------------------------------
    # Prediction value
    # ----------------------------------------------

    prediction = row[
        prediction_column
    ]


    # ----------------------------------------------
    # Accumulation category
    # ----------------------------------------------

    if category_column is not None:

        accumulation = row[
            category_column
        ]

    else:

        accumulation = "Not available"


    # ----------------------------------------------
    # Popup
    # ----------------------------------------------

    popup_html = f"""
    <div style="width:280px">

    <h4>Ocean Plastic Prediction</h4>

    <b>Location</b><br>
    Latitude: {row["Latitude (degree)"]:.4f}<br>
    Longitude: {row["Longitude (degree)"]:.4f}

    <hr>

    <b>Predicted concentration</b><br>
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

    <b>Environmental similarity</b><br>
    Distance: {row["nearest_environment_distance"]:.3f}

    <br><br>

    <b>Environmental support</b><br>
    {row["environment_support"]}

    <br><br>

    <b>Geographic support</b><br>
    {row["geographic_support"]}

    </div>
    """


    folium.CircleMarker(
        location=[
            row["Latitude (degree)"],
            row["Longitude (degree)"]
        ],
        radius=3,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        weight=1,
        popup=folium.Popup(
            popup_html,
            max_width=350
        )
    ).add_to(support_layer)


support_layer.add_to(m)


# ==================================================
# 10. ADD LEGEND
# ==================================================

legend_html = """
<div style="
position: fixed;
bottom: 40px;
left: 40px;
width: 240px;
background-color: white;
border: 2px solid grey;
z-index: 9999;
font-size: 13px;
padding: 12px;
">

<b>Prediction Support</b>

<br><br>

<span style="
background-color:green;
width:12px;
height:12px;
display:inline-block;
margin-right:6px;
"></span>

Supported

<br>

<span style="
background-color:orange;
width:12px;
height:12px;
display:inline-block;
margin-right:6px;
"></span>

Caution

<br>

<span style="
background-color:red;
width:12px;
height:12px;
display:inline-block;
margin-right:6px;
"></span>

Extrapolation

<br><br>

<b>Important:</b>

<br>

Support indicates how well the
prediction location and environmental
conditions are represented by the
training data. It is not a statistical
confidence percentage.

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

output_file = (
    "data/ocean_plastic_prediction_map_support.html"
)

m.save(output_file)


print("\n================================")
print("MAP CREATED")
print("================================")

print(
    "Prediction points:",
    len(df)
)

print(
    "Saved:"
)

print(output_file)