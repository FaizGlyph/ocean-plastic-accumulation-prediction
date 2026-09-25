import pandas as pd
import numpy as np
import xarray as xr
import joblib


# =============================
# SETTINGS
# =============================

DATE = "2019-06-18T05:00:00"

LAT_MIN = 25.25
LAT_MAX = 40.75

LON_MIN = -156.25
LON_MAX = -130.25

GRID_STEP = 0.25


# =============================
# HELPER
# =============================

def nearest_indices(values, targets):
    values = np.asarray(values)
    targets = np.asarray(targets)

    indices = np.searchsorted(values, targets)

    indices = np.clip(
        indices,
        1,
        len(values) - 1
    )

    left = values[indices - 1]
    right = values[indices]

    choose_right = (
        np.abs(right - targets)
        <
        np.abs(targets - left)
    )

    return np.where(
        choose_right,
        indices,
        indices - 1
    )


# =============================
# LOAD MODEL
# =============================

saved = joblib.load(
    "model/random_forest_model.joblib"
)

model = saved["model"]
features = saved["features"]


# =============================
# CREATE GRID
# =============================

latitudes = np.arange(
    LAT_MIN,
    LAT_MAX + GRID_STEP,
    GRID_STEP
)

longitudes = np.arange(
    LON_MIN,
    LON_MAX + GRID_STEP,
    GRID_STEP
)

grid = pd.DataFrame(
    [
        (lat, lon)
        for lat in latitudes
        for lon in longitudes
    ],
    columns=[
        "Latitude (degree)",
        "Longitude (degree)"
    ]
)

print("Grid points:", len(grid))


# =============================
# LOAD DATA
# =============================

wind = xr.open_dataset(
    "data/ERA5/ERA5_wind.nc"
)

waves = xr.open_dataset(
    "data/ERA5/ERA5_waves.nc"
)

current = xr.open_dataset(
    "data/currents/GLOBCURRENT_2015_2019.nc"
)


# =============================
# SORT COORDINATES
# =============================

wind = wind.sortby("latitude").sortby("longitude")
waves = waves.sortby("latitude").sortby("longitude")
current = current.sortby("latitude").sortby("longitude")


# =============================
# GRID COORDINATES
# =============================

lat_target = grid[
    "Latitude (degree)"
].values

lon_target = grid[
    "Longitude (degree)"
].values


# =============================
# WIND
# =============================

wind_day = wind.sel(
    valid_time=pd.Timestamp(DATE),
    method="nearest"
)

wind_lat_idx = nearest_indices(
    wind.latitude.values,
    lat_target
)

wind_lon_idx = nearest_indices(
    wind.longitude.values,
    lon_target
)

wind_u = wind_day["u10"].values[
    wind_lat_idx,
    wind_lon_idx
]

wind_v = wind_day["v10"].values[
    wind_lat_idx,
    wind_lon_idx
]

grid["wind_u"] = wind_u
grid["wind_v"] = wind_v

grid["wind_speed"] = np.sqrt(
    wind_u ** 2 +
    wind_v ** 2
)


# =============================
# WAVES
# =============================

wave_day = waves.sel(
    valid_time=pd.Timestamp(DATE),
    method="nearest"
)

wave_lat_idx = nearest_indices(
    waves.latitude.values,
    lat_target
)

wave_lon_idx = nearest_indices(
    waves.longitude.values,
    lon_target
)

wave_direction = wave_day["mwd"].values[
    wave_lat_idx,
    wave_lon_idx
]

wave_period = wave_day["mwp"].values[
    wave_lat_idx,
    wave_lon_idx
]

wave_height = wave_day["swh"].values[
    wave_lat_idx,
    wave_lon_idx
]

grid["wave_direction"] = wave_direction
grid["wave_period"] = wave_period
grid["wave_height"] = wave_height


# =============================
# CURRENTS
# =============================

current_day = current.sel(
    time=pd.Timestamp(DATE).normalize(),
    method="nearest"
).isel(depth=0)

current_lat_idx = nearest_indices(
    current.latitude.values,
    lat_target
)

current_lon_idx = nearest_indices(
    current.longitude.values,
    lon_target
)

current_u = current_day["uo"].values[
    current_lat_idx,
    current_lon_idx
]

current_v = current_day["vo"].values[
    current_lat_idx,
    current_lon_idx
]

grid["current_u"] = current_u
grid["current_v"] = current_v

grid["current_speed"] = np.sqrt(
    current_u ** 2 +
    current_v ** 2
)


# =============================
# MONTH
# =============================

grid["month"] = pd.Timestamp(DATE).month


# =============================
# CHECK DATA
# =============================

environmental_columns = [
    "wind_u",
    "wind_v",
    "wind_speed",
    "wave_direction",
    "wave_period",
    "wave_height",
    "current_u",
    "current_v",
    "current_speed"
]

print("\nMissing values:")
print(
    grid[environmental_columns]
    .isna()
    .sum()
)


# =============================
# PREDICTION
# =============================

X = grid[features]

grid["Predicted_Concentration"] = (
    model.predict(X)
)


# =============================
# ACCUMULATION POTENTIAL
# =============================

grid["Accumulation_Potential"] = np.select(
    [
        grid["Predicted_Concentration"] < 1,
        grid["Predicted_Concentration"] < 10
    ],
    [
        "Low",
        "Medium"
    ],
    default="High"
)


# =============================
# SAVE
# =============================

grid.to_csv(
    "data/prediction_grid.csv",
    index=False
)


# =============================
# RESULTS
# =============================

print("\nPREDICTION COMPLETE")
print("-------------------")

print(
    "Minimum prediction:",
    round(
        grid["Predicted_Concentration"].min(),
        3
    )
)

print(
    "Maximum prediction:",
    round(
        grid["Predicted_Concentration"].max(),
        3
    )
)

print("\nAccumulation Potential:")
print(
    grid["Accumulation_Potential"]
    .value_counts()
)

print("\nSaved:")
print("data/prediction_grid.csv")