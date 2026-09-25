import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import BallTree, NearestNeighbors


# ==================================================
# 1. LOAD DATA
# ==================================================

df = pd.read_csv("data/ml_dataset.csv")

print("Dataset:", df.shape)


# ==================================================
# 2. FEATURES AND TARGET
# ==================================================

features = [
    "wind_u",
    "wind_v",
    "wind_speed",
    "wave_direction",
    "wave_period",
    "wave_height",
    "current_u",
    "current_v",
    "current_speed",
    "month",
    "Latitude (degree)",
    "Longitude (degree)"
]

environmental_features = [
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


# ==================================================
# 3. CREATE 2° SPATIAL BLOCKS
# ==================================================

BLOCK_SIZE = 2.0

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

groups = df["spatial_block"]


# ==================================================
# 4. SPATIAL 5-FOLD CV
# ==================================================

gkf = GroupKFold(n_splits=5)

all_results = []


for fold, (train_idx, test_idx) in enumerate(
    gkf.split(X, y, groups=groups),
    start=1
):

    print("\n================================")
    print(f"FOLD {fold}")
    print("================================")


    # ----------------------------------------------
    # Split data
    # ----------------------------------------------

    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()


    X_train = train[features]
    X_test = test[features]

    y_train = train[target]
    y_test = test[target]


    # ----------------------------------------------
    # Train Random Forest
    # ----------------------------------------------

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)


    # ----------------------------------------------
    # Calculate prediction errors
    # ----------------------------------------------

    errors = np.abs(
        y_test.values - predictions
    )


    # ==================================================
    # GEOGRAPHIC SUPPORT
    # ==================================================

    train_coords = np.radians(
        train[
            ["Latitude (degree)",
             "Longitude (degree)"]
        ].values
    )

    test_coords = np.radians(
        test[
            ["Latitude (degree)",
             "Longitude (degree)"]
        ].values
    )


    tree = BallTree(
        train_coords,
        metric="haversine"
    )

    distances, _ = tree.query(
        test_coords,
        k=1
    )

    EARTH_RADIUS_KM = 6371.0

    geo_distance = (
        distances[:, 0]
        * EARTH_RADIUS_KM
    )


    # ==================================================
    # ENVIRONMENTAL SUPPORT
    # ==================================================

    scaler = StandardScaler()

    train_env = scaler.fit_transform(
        train[environmental_features]
    )

    test_env = scaler.transform(
        test[environmental_features]
    )


    nn = NearestNeighbors(
        n_neighbors=1,
        metric="euclidean"
    )

    nn.fit(train_env)

    env_distances, _ = nn.kneighbors(
        test_env
    )

    env_distance = env_distances[:, 0]


    # ==================================================
    # ENVIRONMENTAL THRESHOLDS
    # ==================================================
    #
    # Calculate the distribution of environmental
    # distances among the TRAINING observations.
    #
    # This avoids using the test data to define
    # the support thresholds.
    # ==================================================

    train_nn = NearestNeighbors(
        n_neighbors=2,
        metric="euclidean"
    )

    train_nn.fit(train_env)

    train_distances, _ = train_nn.kneighbors(
        train_env
    )

    # First neighbour is the point itself,
    # so use the second neighbour.

    train_environment_distance = (
        train_distances[:, 1]
    )

    env_90 = np.percentile(
        train_environment_distance,
        90
    )

    env_95 = np.percentile(
        train_environment_distance,
        95
    )


    # ==================================================
    # SUPPORT CLASSIFICATION
    # ==================================================

    support_categories = []

    for geo, env in zip(
        geo_distance,
        env_distance
    ):

        if (
            geo > 200
            or
            env > env_95
        ):

            category = "Extrapolation"

        elif (
            geo > 100
            or
            env > env_90
        ):

            category = "Caution"

        else:

            category = "Supported"


        support_categories.append(category)


    # ==================================================
    # FOLD SUMMARY
    # ==================================================

    fold_df = pd.DataFrame({

        "fold": fold,

        "actual": y_test.values,

        "predicted": predictions,

        "absolute_error": errors,

        "geo_distance_km":
            geo_distance,

        "environment_distance":
            env_distance,

        "support":
            support_categories
    })


    # ----------------------------------------------
    # Print fold performance
    # ----------------------------------------------

    print(
        "MAE:",
        round(
            mean_absolute_error(
                y_test,
                predictions
            ),
            4
        )
    )

    print(
        "R²:",
        round(
            r2_score(
                y_test,
                predictions
            ),
            4
        )
    )


    print(
        "\nSupport distribution:"
    )

    print(
        fold_df["support"]
        .value_counts()
    )


    # ----------------------------------------------
    # Error by support category
    # ----------------------------------------------

    print(
        "\nMAE by support:"
    )

    for category in [
        "Supported",
        "Caution",
        "Extrapolation"
    ]:

        subset = fold_df[
            fold_df["support"] == category
        ]

        if len(subset) > 0:

            print(
                f"{category:15}: "
                f"{len(subset):3} samples | "
                f"MAE = "
                f"{subset['absolute_error'].mean():.4f}"
            )


    all_results.append(
        fold_df
    )


# ==================================================
# 5. COMBINE ALL FOLDS
# ==================================================

results = pd.concat(
    all_results,
    ignore_index=True
)


# ==================================================
# 6. FINAL ANALYSIS
# ==================================================

print("\n\n================================")
print("SUPPORT vs PREDICTION ERROR")
print("================================")


print(
    "\nOverall spatial CV MAE:",
    round(
        results["absolute_error"].mean(),
        4
    )
)


print(
    "\nSamples by support:"
)

print(
    results["support"]
    .value_counts()
)


print(
    "\nMAE by support category:"
)


for category in [
    "Supported",
    "Caution",
    "Extrapolation"
]:

    subset = results[
        results["support"] == category
    ]

    if len(subset) > 0:

        print(
            f"{category:15}: "
            f"{len(subset):3} samples | "
            f"MAE = "
            f"{subset['absolute_error'].mean():.4f}"
        )


# ==================================================
# 7. SAVE RESULTS
# ==================================================

output_file = (
    "data/spatial_cv_support_results.csv"
)

results.to_csv(
    output_file,
    index=False
)


print(
    "\nSaved:"
)

print(output_file)