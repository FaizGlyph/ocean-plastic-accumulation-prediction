# Ocean Plastic Accumulation Zone Prediction Model

## Machine Learning Based Prediction and GIS Visualization

### Mini Project — CSE7102

**Program:** B.Tech Computer Science and Engineering  
**Semester:** VII  
**Institution:** Presidency University, Bengaluru

---

## 1. Project Overview

Marine plastic does not spread randomly across the ocean. Its distribution and accumulation are influenced by factors such as ocean currents, wind, waves, geographical location and time.

This project proposes a machine-learning-based system to predict potential ocean plastic accumulation zones using historical plastic observations and environmental/oceanographic data.

The predicted results will be visualized through an interactive GIS-based web interface.

---

## 2. Problem Statement

Existing marine plastic observations are fragmented and difficult to use directly for identifying potential accumulation areas.

The proposed system aims to combine historical plastic observations with environmental and oceanographic information and use machine learning to estimate plastic concentration for selected ocean locations or regions.

The predictions will then be displayed on an interactive map to make potential accumulation areas easier to identify and study.

---

## 3. Objectives

- Collect and prepare historical marine plastic observation data.
- Identify suitable environmental and oceanographic variables.
- Combine and preprocess data from compatible sources.
- Develop and evaluate machine learning models for plastic concentration prediction.
- Visualize predicted concentrations using GIS.
- Provide a simple web-based interface for exploring prediction results.

---

## 4. Proposed Inputs

The initial project will focus on a manageable set of inputs:

- Historical plastic observations
- Ocean current data
- Wind data
- Wave conditions
- Geographic coordinates
- Date/season information

Additional variables may be considered only if they are required and compatible with the available data.

---

## 5. Proposed Methodology

The project follows the workflow:

**Data Collection**  
↓  
**Data Preprocessing**  
↓  
**Feature Engineering**  
↓  
**ML Model Training**  
↓  
**Plastic Concentration Prediction**  
↓  
**GIS Visualization**

### Model Training

Model training will be performed offline using historical labelled data.

Different suitable machine learning models may be compared, and the final model will be selected based on its performance and suitability for the available dataset.

---

## 6. Expected Output

The final system is expected to provide:

- Predicted plastic concentration for selected ocean locations/regions.
- Identification of potential accumulation hotspots.
- Interactive GIS visualization of predictions.
- A simple web interface for exploring locations and prediction results.

The outputs are proposed at this stage. Actual prediction results will be generated after model training and validation.

---

## 7. Proposed Technology Stack

### Data & Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn

### Backend

- FastAPI

### Frontend

- React

### Visualization

- GIS mapping library

### Development & Version Control

- Jupyter Notebook / VS Code
- Git
- GitHub

---

## 8. Project Structure

```text
ocean-plastic-accumulation-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│   ├── preprocessing/
│   ├── features/
│   └── models/
│
├── backend/
│
├── frontend/
│
├── docs/
│
├── README.md
└── requirements.txt
