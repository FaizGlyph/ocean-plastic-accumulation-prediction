import pandas as pd
import xarray as xr
import numpy as np

# =============================
# FILES
# =============================
plastic_file = "data/plastic/original_plastic_data.csv"
wind_file = "data/ERA5/ERA5_wind.nc"
waves_file = "data/ERA5/ERA5_waves.nc"
current_file = "data/currents/GLOBCURRENT_2015_2019.nc"

# =============================
# 1. LOAD PLASTIC DATA
# =============================
plastic = pd.read_csv(plastic_file)

plastic["datetime"] = pd.to_datetime(
    plastic["Sample Date"],
    errors="coerce"
)

print("Plastic observations:", len(plastic))

# =============================
# 2. LOAD ENVIRONMENTAL DATA
# =============================
wind = xr.open_dataset(wind_file)
waves = xr.open_dataset(waves_file)
current = xr.open_dataset(current_file)

# Make coordinates increasing
wind = wind.sortby("latitude").sortby("longitude")
waves = waves.sortby("latitude").sortby("longitude")
current = current.sortby("latitude").sortby("longitude")

# =============================
# 3. ERA5 WIND
# =============================
wind_values = wind[["u10", "v10"]].interp(
    valid_time=xr.DataArray(
        plastic["datetime"].values,
        dims="obs"
    ),
    latitude=xr.DataArray(
        plastic["Latitude (degree)"].values,
        dims="obs"
    ),
    longitude=xr.DataArray(
        plastic["Longitude (degree)"].values,
        dims="obs"
    )
)

plastic["wind_u"] = wind_values["u10"].values
plastic["wind_v"] = wind_values["v10"].values

plastic["wind_speed"] = np.sqrt(
    plastic["wind_u"] ** 2 +
    plastic["wind_v"] ** 2
)

# =============================
# 4. ERA5 WAVES
# =============================
wave_values = waves[["mwd", "mwp", "swh"]].interp(
    valid_time=xr.DataArray(
        plastic["datetime"].values,
        dims="obs"
    ),
    latitude=xr.DataArray(
        plastic["Latitude (degree)"].values,
        dims="obs"
    ),
    longitude=xr.DataArray(
        plastic["Longitude (degree)"].values,
        dims="obs"
    )
)

plastic["wave_direction"] = wave_values["mwd"].values
plastic["wave_period"] = wave_values["mwp"].values
plastic["wave_height"] = wave_values["swh"].values

# =============================
# 5. GLOBCURRENT
# =============================
plastic["current_date"] = plastic["datetime"].dt.normalize()

current_values = current[["uo", "vo"]].sel(
    time=xr.DataArray(
        plastic["current_date"].values,
        dims="obs"
    ),
    latitude=xr.DataArray(
        plastic["Latitude (degree)"].values,
        dims="obs"
    ),
    longitude=xr.DataArray(
        plastic["Longitude (degree)"].values,
        dims="obs"
    ),
    method="nearest"
)

plastic["current_u"] = current_values["uo"].values
plastic["current_v"] = current_values["vo"].values

plastic["current_speed"] = np.sqrt(
    plastic["current_u"] ** 2 +
    plastic["current_v"] ** 2
)
# =============================
# 6. TIME FEATURE
# =============================
plastic["month"] = plastic["datetime"].dt.month

# =============================
# 7. CHECK MISSING VALUES
# =============================
features = [
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

print("\nMissing environmental values:")
print(plastic[features].isna().sum())

print("\nRows:", len(plastic))

# =============================
# 8. CREATE ML DATASET
# =============================
output_columns = [
    "Unique ID",
    "Sample Date",
    "Latitude (degree)",
    "Longitude (degree)",
    "Microplastics Measurement",
    "Unit",
    "Concentration Class",
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

ml_data = plastic[output_columns]

# =============================
# 9. SAVE
# =============================
ml_data.to_csv(
    "data/ml_dataset.csv",
    index=False
)

print("\nDONE!")
print("Saved: data/ml_dataset.csv")