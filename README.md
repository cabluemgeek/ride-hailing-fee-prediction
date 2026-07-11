# **Ride-Hailing Trip Fee Prediction**

Predicting trip fees for an Algerian ride-hailing platform using historical trip, driver, rider and weather data. Built as an end-to-end data science project: EDA → customer segmentation → geospatial analysis → feature engineering → ML modeling.

## Problem

Given trip-level data (pickup/destination city, distances, timestamps, rider/driver info, discounts, cancellations), predict `trip_fee` for unseen trips. The target is evaluated on completed trips only, so a large share of the work is understanding *why* trips fail (cancellations, unaccepted requests) before ever touching the regression problem.

## Dataset

Trip records from a ride-hailing service operating in 47 Algerian cities. Each row is one trip request with:

- Timestamps: `request_date`, `accepted_date`, `started_date`, `trip_finished_date`
- Trip info: `pickup_city`, `destination_city`, `trip_distance`, `driver2rider_distance`, `trip_status`
- People: `rider`, `driver`, `rider_rating`, `driver_rating`
- Money: `used_discount`, `discount_amount`, `trip_fee` (target)
- Cancellation reasons (rider/driver side)

Missing values are concentrated in trips that were never accepted, started, or completed (driver info, ratings, fees) — these are treated as **structurally missing**, not data errors, and handled accordingly rather than dropped.

## Approach

**1. Descriptive statistics** — finished vs. unfinished trip rates broken down by month, time of day, and city, to understand where and when demand is lost.

[![View Dashboard](https://img.shields.io/badge/View-Live%20Dashboard-blue)](https://cabluemgeek.github.io/ride-hailing-fee-prediction/ride_hailing_dashboard.html)

**2. Exploratory data analysis**
- *Customer segmentation* (K-Means on frequency, completion rate, cancellation rate, distance, spend) → 4 rider profiles. The key finding: the highest-spending, highest-frequency segment also has the worst completion rate — a retention risk sitting on top of the platform's best revenue.
- *Driver-level KPIs*
- *Interactive geographic analysis* (Folium) mapping demand, cancellation, and revenue flows between pickup/destination cities to surface high-demand routes and problem areas.

**3. Feature engineering**
- Time features (`hour`, `day of week`) from request timestamp
- `same_city` flag, trip duration in seconds (with a missingness flag — some trips marked `FINISHED` had implausible <1-minute durations, likely drivers mis-marking trip status)
- Discount amount imputation
- **Weather enrichment**: pickup coordinates + trip date sent to the [Open-Meteo](https://open-meteo.com/) archive API, hourly weather codes collapsed to a daily condition (Sunny/Cloudy/Fog/Drizzle/Rainy/…). Calls are batched on unique (city, date) pairs rather than per-row to keep API usage sane.

**4. Modeling** — trained and compared 5 regressors on log-transformed fee:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Random Forest** | **52.32** | 387.82 | **0.9496** |
| Ensemble (avg of XGB/LGBM/CatBoost) | 53.85 | 435.19 | 0.9365 |
| LightGBM | 54.23 | 438.61 | 0.9355 |
| CatBoost | 54.91 | 419.35 | 0.9411 |
| XGBoost | 55.23 | 482.56 | 0.9219 |
| Deep Learning (MLP) | 106.74 | 1044.52 | 0.6342 |

A small feedforward neural net (Keras/TensorFlow) was also tested as a baseline comparison against the tree-based models. It underperformed by a wide margin (R² 0.63 vs. ~0.94-0.95 for the tree models) — a useful result in itself: on this tabular, mixed-type feature set, tree ensembles capture the signal far more efficiently than a plain MLP, without needing embedding layers or extensive tuning.

Random Forest was selected for the final submission based on lowest MAE.

## Tech stack

`Python` · `pandas` · `numpy` · `scikit-learn` · `XGBoost` · `LightGBM` · `CatBoost` · `TensorFlow/Keras` · `Folium` · `Plotly` · `Matplotlib/Seaborn` · `geopy` · Open-Meteo API

## Repository structure

```
ride-hailing-fee-prediction/
├── .gitignore
├── README.md
├── requirements.txt
├── ride_hailing_analysis.ipynb
└── images/              # exported charts/maps referenced in this README
```

## Key insights

- A frequent, high-spending rider segment has a disproportionately high cancellation rate — a targeted retention opportunity rather than a pure "grow demand" problem.
- Trip failure (not fee amount) is the biggest lever on revenue: a large share of requests never reach `FINISHED`.
- Weather condition, time of day, and same-city vs. cross-city routing all carry real signal for fee prediction once combined with distance features.

## Possible next steps

- Hyperparameter tuning (Optuna) on the Random Forest / LightGBM models
- SHAP-based feature importance for the final model
- Try stacking instead of simple averaging for the ensemble
