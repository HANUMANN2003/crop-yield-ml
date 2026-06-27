# 🌾 Smart Crop Yield Prediction System

## Overview

The Smart Crop Yield Prediction System is a Machine Learning-powered web application that helps estimate agricultural crop yield based on environmental, climatic, and farming-related parameters. The project combines data analysis, predictive modeling, and an interactive Streamlit dashboard to provide farmers, researchers, and agricultural planners with data-driven insights.

The system uses machine learning models to predict crop yield and classify expected yield performance using factors such as rainfall, temperature, humidity, irrigation access, soil conditions, and crop characteristics.

---

## Features

### 📊 Crop Yield Prediction

Predict expected crop yield using trained machine learning models.

### 🌦 Climate-Based Analysis

Analyze the impact of environmental factors such as:

* Temperature
* Rainfall
* Humidity

### 🌱 Agricultural Insights

Understand how different farming conditions influence crop productivity.

### 📈 Interactive Dashboard

User-friendly Streamlit interface for:

* Inputting crop parameters
* Viewing predictions
* Exploring visualizations
* Analyzing model performance

### 🔍 Exploratory Data Analysis (EDA)

Interactive charts and visualizations to identify:

* Crop trends
* Yield distribution
* Environmental impacts
* Feature relationships

### 🤖 Machine Learning Models

The project includes:

* HistGradientBoosting Regressor for crop yield prediction
* HistGradientBoosting Classifier for yield category classification

---

## Project Structure

```text
crop-yield-ml/
│
├── app.py                      # Streamlit application
├── Crop_Yield.csv              # Dataset
├── crop_yield_model.pkl        # Trained regression model
├── yield_classifier.pkl        # Trained classification model
├── label_encoders.pkl          # Saved label encoders
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

---

## Machine Learning Workflow

### 1. Data Collection

Agricultural and environmental data collected from the dataset.

### 2. Data Preprocessing

* Handling missing values
* Encoding categorical variables
* Data cleaning

### 3. Feature Engineering

Additional features created to improve prediction performance:

* Rain_per_Day
* Climate_Index

### 4. Model Training

The dataset is split into training and testing sets.

Models trained:

* HistGradientBoosting Regressor
* HistGradientBoosting Classifier

### 5. Model Evaluation

Performance evaluated using:

#### Regression Metrics

* R² Score
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

#### Classification Metrics

* Accuracy
* Precision
* Recall
* F1 Score

### 6. Deployment

Models are integrated into a Streamlit dashboard for real-time predictions.

---

## Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Plotly
* Streamlit
* Joblib

---

## Installation

### Clone Repository

```bash
git clone https://github.com/HANUMANN2003/crop-yield-ml.git
cd crop-yield-ml
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate:

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

## Example Workflow

1. Launch the Streamlit application.
2. Enter crop and environmental parameters.
3. Click Predict.
4. View:

   * Predicted Yield
   * Yield Classification
   * Data Visualizations
   * Model Insights

---

## Future Enhancements

Planned improvements include:

* Farm Area (Acres/Hectares) Support
* Crop Recommendation System
* Weather API Integration
* Fertilizer Recommendation Engine
* Profit Forecasting
* SHAP-Based Explainable AI
* PDF Report Generation
* Interactive Geographic Yield Maps
* Model Comparison Dashboard

---

## Results

The machine learning models successfully predict crop yield and classify yield performance based on agricultural and climatic factors. The interactive dashboard provides a practical decision-support tool for agricultural planning and productivity analysis.

---

## Author

**Hanuman**

GitHub: https://github.com/HANUMANN2003

---

## License

This project is intended for educational and research purposes.
