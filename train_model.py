import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("data/ml_dataset.csv")

# Features
features = [
    "Latitude (degree)",
    "Longitude (degree)",
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

target = "Microplastics Measurement"

X = df[features]
y = df[target]

# Train final model using all observations
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

# Save model
joblib.dump(
    {
        "model": model,
        "features": features
    },
    "model/random_forest_model.joblib"
)

print("FINAL MODEL TRAINED")
print("-------------------")
print("Training samples:", len(df))
print("Features:", len(features))
print("Model saved to:")
print("model/random_forest_model.joblib")