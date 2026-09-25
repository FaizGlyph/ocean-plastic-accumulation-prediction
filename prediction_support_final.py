import pandas as pd
import numpy as np


# --------------------------------------------------
# 1. Load the prediction grid
# --------------------------------------------------

grid = pd.read_csv(
    "data/prediction_grid_support.csv"
)

print("Prediction grid:", len(grid))


# --------------------------------------------------
# 2. Environmental distance thresholds
# --------------------------------------------------

environment_90 = np.percentile(
    grid["nearest_environment_distance"],
    90
)

environment_95 = np.percentile(
    grid["nearest_environment_distance"],
    95
)


# --------------------------------------------------
# 3. Environmental support
# --------------------------------------------------

def classify_environment(distance):

    if distance <= environment_90:
        return "Familiar"

    elif distance <= environment_95:
        return "Caution"

    else:
        return "Unusual"


grid["environment_support"] = (
    grid["nearest_environment_distance"]
    .apply(classify_environment)
)


# --------------------------------------------------
# 4. Combined prediction support
# --------------------------------------------------

def classify_combined(row):

    geographic_distance = (
        row["nearest_observation_km"]
    )

    environment_distance = (
        row["nearest_environment_distance"]
    )

    # Strongest warning:
    # geographically far OR environmentally unusual

    if (
        geographic_distance > 200
        or
        environment_distance > environment_95
    ):
        return "Extrapolation"

    # Moderate warning:
    # geographically farther OR environmental conditions
    # are somewhat unusual

    elif (
        geographic_distance > 100
        or
        environment_distance > environment_90
    ):
        return "Caution"

    else:
        return "Supported"


grid["prediction_support"] = (
    grid.apply(
        classify_combined,
        axis=1
    )
)


# --------------------------------------------------
# 5. Save
# --------------------------------------------------

output_file = (
    "data/prediction_grid_support.csv"
)

grid.to_csv(
    output_file,
    index=False
)


# --------------------------------------------------
# 6. Print results
# --------------------------------------------------

print("\n================================")
print("COMBINED PREDICTION SUPPORT")
print("================================")

print(
    "\nEnvironmental thresholds:"
)

print(
    "90th percentile:",
    round(environment_90, 4)
)

print(
    "95th percentile:",
    round(environment_95, 4)
)


print("\nEnvironmental support:")

env_counts = (
    grid["environment_support"]
    .value_counts()
)

env_percent = (
    grid["environment_support"]
    .value_counts(normalize=True)
    * 100
)

for category in [
    "Familiar",
    "Caution",
    "Unusual"
]:

    print(
        f"{category:10}: "
        f"{env_counts.get(category, 0):5} "
        f"points "
        f"({env_percent.get(category, 0):.1f}%)"
    )


print("\nFinal prediction support:")

support_counts = (
    grid["prediction_support"]
    .value_counts()
)

support_percent = (
    grid["prediction_support"]
    .value_counts(normalize=True)
    * 100
)

for category in [
    "Supported",
    "Caution",
    "Extrapolation"
]:

    print(
        f"{category:15}: "
        f"{support_counts.get(category, 0):5} "
        f"points "
        f"({support_percent.get(category, 0):.1f}%)"
    )


print("\nSaved:")
print(output_file)