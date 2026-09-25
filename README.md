# Ocean Plastic Accumulation Zone Prediction Model

## Machine Learning Based Prediction and GIS Visualization

**Mini Project — CSE7102**  
**B.Tech Computer Science and Engineering**  
**Presidency University, Bengaluru**

### Team

- Faizan Ahmed Bagalkot
- Mohammed Khubaib
- Danish Mehdi

**Project Guide:** Ms. Rama Bai V.

---

## 1. Project Overview

Marine plastic pollution is influenced by environmental and oceanographic conditions such as wind, waves, and ocean currents. Historical plastic observations, however, are limited to specific sampling locations and dates.

This project develops a machine-learning-based framework that combines historical plastic observations with environmental and oceanographic data to estimate plastic concentration and visualize spatial accumulation potential across a defined Pacific Ocean study region.

The system uses a Random Forest Regression model to learn relationships between environmental conditions and observed plastic concentration. The trained model is then applied over a spatial prediction grid and the results are visualized using an interactive GIS interface called OceanTrace.

The system provides model-based predictions and does not directly detect plastic from satellite imagery or provide real-time plastic detection.

---

## 2. Project Objectives

The main objectives are:

1. Integrate historical plastic observations with environmental and oceanographic data.
2. Prepare a machine-learning dataset using spatial and temporal matching.
3. Train a Random Forest regression model to estimate plastic concentration.
4. Evaluate the model using random and spatial cross-validation.
5. Generate predictions over a spatial grid covering the study region.
6. Identify areas where predictions are relatively supported by the training data and areas requiring caution.
7. Visualize the results through an interactive GIS-based interface.

---

## 3. Data Sources

### Historical Plastic Observations

The project uses 614 historical marine plastic observations.

The observations cover:

- Period: 2015–2019
- Region: Pacific Ocean study region
- Target variable: Microplastic concentration
- Unit: pieces/m³
- Latitude and longitude of each observation

### ERA5 Reanalysis Data

ERA5 environmental data provides:

- 10 m u-component wind
- 10 m v-component wind
- Mean wave direction
- Mean wave period
- Significant wave height

### GLOBCURRENT Ocean Current Data

GLOBCURRENT provides:

- Zonal current component (uo)
- Meridional current component (vo)
- Daily surface current information

These datasets are spatially and temporally matched with the historical plastic observations.

---

## 4. Data Processing

The environmental variables are matched to each historical plastic observation using the observation location and date/time.

Feature engineering produces the final machine-learning inputs.

Derived variables include:

- Wind Speed = square root of (wind_u² + wind_v²)
- Current Speed = square root of (current_u² + current_v²)
- Month = month extracted from the observation date

The resulting machine-learning dataset contains:

- 614 observations
- 12 input features
- 1 target variable

### Input Features

1. Wind u
2. Wind v
3. Wind speed
4. Wave direction
5. Wave period
6. Wave height
7. Current u
8. Current v
9. Current speed
10. Month
11. Latitude
12. Longitude

### Target

Microplastic concentration (pieces/m³)

---

## 5. Machine Learning

The project uses a Random Forest Regression model.

The model learns the relationship between the 12 input features and observed microplastic concentration.

The trained model is saved as:

model/random_forest_model.joblib

---

## 6. Model Evaluation

Two validation approaches were used.

### Random 5-Fold Cross-Validation

Mean results across five folds:

- MAE: 1.9246
- R²: 0.5264

### Spatial Cross-Validation

A spatial-block validation approach using 2° × 2° geographic blocks was also performed to examine geographic generalization.

Mean results:

- MAE: 2.7195
- R²: -0.0939

The difference between random and spatial validation indicates that geographic generalization is more difficult than random validation suggests.

Therefore, the spatial validation results are reported separately rather than presenting the random cross-validation result as the only measure of model performance.

---

## 7. Spatial Prediction

The trained Random Forest model was applied over a spatial prediction grid containing 6,615 prediction points.

The model generated predicted plastic concentration values for the study region.

For visualization, predictions were grouped into accumulation-potential categories:

| Category | Grid Points |
| --- | ---: |
| Low | 570 |
| Medium | 5,643 |
| High | 402 |

These categories represent model predictions, not direct observations of plastic.

---

## 8. Prediction-Support Analysis

A prediction-support diagnostic was developed to identify areas where model predictions are relatively close to the geographic and environmental conditions represented in the training data.

The prediction grid is classified into:

| Category | Grid Points | Percentage |
| --- | ---: | ---: |
| Supported | 3,869 | 58.5% |
| Caution | 1,776 | 26.8% |
| Extrapolation | 970 | 14.7% |

This is a prediction-support/extrapolation diagnostic, not a statistical confidence estimate.

Validation of the diagnostic showed that extrapolative areas were associated with substantially higher prediction error than supported areas.

---

## 9. GIS Visualization

The project uses GIS-based visualization to display the spatial prediction results.

The final visualization provides:

- Interactive map
- Predicted plastic concentration
- Accumulation-potential categories
- Prediction-support information
- Study-region visualization
- Zoom and pan functionality

The map is implemented as part of the OceanTrace interface.

---

## 10. OceanTrace Interface

OceanTrace is the project interface used to present the research results.

The interface contains three main sections:

### Prediction Map

Displays the model-predicted plastic concentration and accumulation potential across the study region.

### Model Analysis

Displays:

- Random cross-validation metrics
- Spatial cross-validation metrics
- Random Forest feature importance
- Validation interpretation

### Methodology

Explains:

- Historical plastic observations
- ERA5 environmental data
- GLOBCURRENT current data
- Data integration
- Feature engineering
- Random Forest regression
- Spatial prediction
- GIS visualization

---

## 11. Project Structure

    Ocean_Plastic_Project/
    │
    ├── data/
    │   ├── plastic/
    │   │   └── original_plastic_data.csv
    │   │
    │   ├── ERA5/
    │   │   ├── ERA5_wind.nc
    │   │   └── ERA5_waves.nc
    │   │
    │   ├── currents/
    │   │   └── GLOBCURRENT_2015_2019.nc
    │   │
    │   ├── ml_dataset.csv
    │   ├── prediction_grid.csv
    │   ├── prediction_grid_support.csv
    │   └── spatial_cv_support_results.csv
    │
    ├── model/
    │   └── random_forest_model.joblib
    │
    ├── build_ml_dataset.py
    ├── train_model.py
    ├── generate_map_data.py
    ├── create_map.py
    ├── create_map_support.py
    ├── spatial_cv.py
    ├── spatial_cv_diagnostic.py
    ├── prediction_support.py
    ├── prediction_support_final.py
    ├── feature_space_support.py
    ├── support_error_validation.py
    ├── plot_spatial_blocks.py
    │
    └── frontend/
        ├── src/
        ├── public/
        ├── package.json
        └── vite.config.js

---

## 12. Technologies Used

### Programming

- Python
- JavaScript

### Machine Learning

- scikit-learn
- Random Forest Regression
- Cross-validation
- Spatial cross-validation

### Data Processing

- Pandas
- NumPy
- xarray

### GIS / Visualization

- Folium
- Interactive web mapping
- Spatial prediction grids

### Frontend

- React
- Vite
- HTML/CSS/JavaScript

### Development

- Visual Studio Code
- Git
- GitHub

---

## 13. Current Project Status

The current implementation includes:

- Historical plastic dataset integration
- ERA5 wind and wave integration
- GLOBCURRENT current integration
- Final ML dataset generation
- Random Forest regression
- Random cross-validation
- Spatial cross-validation
- Feature-importance analysis
- Spatial prediction over 6,615 grid points
- Prediction-support/extrapolation analysis
- GIS prediction map
- OceanTrace frontend prototype

---

## 14. Important Scope

This project is a research prototype.

The predicted accumulation zones represent model-based estimates derived from historical observations and environmental conditions.

The system does not:

- Directly detect plastic using satellite imagery
- Claim real-time plastic detection
- Simulate the complete physical transport of plastic
- Provide statistical confidence probabilities for individual predictions

The prediction-support layer is provided to communicate where predictions are closer to or farther from the geographic and environmental conditions represented in the training data.

---

## 15. Repository

This repository contains the implementation and supporting files for the CSE7102 mini project:

Ocean Plastic Accumulation Zone Prediction Model — Machine Learning Based Prediction and GIS Visualization
