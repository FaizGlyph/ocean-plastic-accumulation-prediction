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

The system uses a **Random Forest Regression** model to learn relationships between environmental conditions and observed plastic concentration. The trained model is then applied over a spatial prediction grid and the results are visualized using an interactive GIS interface called **OceanTrace**.

The system provides model-based predictions and does **not** directly detect plastic from satellite imagery or provide real-time plastic detection.

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

The project uses **614 historical marine plastic observations**.

The observations cover:

- Period: **2015–2019**
- Region: Pacific Ocean study region
- Target variable: Microplastic concentration
- Unit: **pieces/m³**
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

- Zonal current component (`uo`)
- Meridional current component (`vo`)
- Daily surface current information

These datasets are spatially and temporally matched with the historical plastic observations.

---

## 4. Data Processing

The environmental variables are matched to each historical plastic observation using the observation location and date/time.

Feature engineering produces the final machine-learning inputs.

Derived variables include:

```text
Wind Speed = √(wind_u² + wind_v²)

Current Speed = √(current_u² + current_v²)

Month = month extracted from the observation date