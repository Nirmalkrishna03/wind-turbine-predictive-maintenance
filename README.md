# Predictive Maintenance of Wind Turbine Systems using AI

This repository contains the notebooks for predictive maintenance of wind turbine systems using AI. The project applies data analysis and machine learning to wind turbine SCADA data to explore predictive maintenance: detecting unusual behaviour, classifying turbine status and events, and estimating Remaining Useful Life (RUL).

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Objectives](#objectives)
- [Approach](#approach)
- [Notebooks](#notebooks)
- [Modeling Techniques](#modeling-techniques)
- [Results](#results)
- [Requirements](#requirements)

## Project Overview

Unplanned turbine downtime is costly, so early warning of faults helps schedule maintenance. This project explores how far SCADA sensor data can support that, using exploratory analysis, anomaly detection, feature selection and a range of classical and deep-learning models.

## Dataset

The notebooks use real SCADA data from five onshore wind turbines in Portugal, with about 1.2 million records. It includes operating signals such as wind speed, power output and reactive power, plus status codes and event descriptions.

The raw data is not included in this repository.

## Objectives

- Explore SCADA signals and their relationships.
- Detect anomalous behaviour in turbine data.
- Select a compact set of informative features.
- Classify turbine status and event types.
- Estimate Remaining Useful Life (RUL) with hybrid deep-learning models.
- Model reactive power from the other SCADA signals.

## Approach

1. **Data preprocessing:** cleaning, scaling and preparing the data.
2. **Exploratory data analysis:** distributions, correlations and anomalies.
3. **Feature selection:** seven selection methods, keeping features chosen by at least four of them.
4. **Model development:** classical machine-learning and deep-learning models for classification, RUL estimation and regression.
5. **Evaluation:** metrics on held-out data where a split was used.

## Notebooks

| Notebook | Description |
|---|---|
| [`01_basic_eda.ipynb`](notebooks/01_basic_eda.ipynb) | Initial data exploration and visualisation |
| [`02_data_preprocessing.ipynb`](notebooks/02_data_preprocessing.ipynb) | Cleaning, scaling and baseline classifiers |
| [`03_correlation_analysis.ipynb`](notebooks/03_correlation_analysis.ipynb) | Correlation heatmaps and Granger causality tests |
| [`04_anomaly_detection.ipynb`](notebooks/04_anomaly_detection.ipynb) | Z-score, Isolation Forest and other anomaly detection methods |
| [`05_feature_selection.ipynb`](notebooks/05_feature_selection.ipynb) | Feature selection with seven methods and majority voting |
| [`06_failure_types.ipynb`](notebooks/06_failure_types.ipynb) | Failure-type classification |
| [`07_local_event_description.ipynb`](notebooks/07_local_event_description.ipynb) | Event-description classification with several model families |
| [`08_rul_prediction.ipynb`](notebooks/08_rul_prediction.ipynb) | RUL estimation with hybrid architectures |
| [`09_reactive_power_management.ipynb`](notebooks/09_reactive_power_management.ipynb) | Reactive power regression |

`01_basic_eda.ipynb` is large and may not render on GitHub; open it through [nbviewer](https://nbviewer.org/github/Nirmalkrishna03/wind-turbine-predictive-maintenance/blob/main/notebooks/01_basic_eda.ipynb) instead.

## Modeling Techniques

- **Anomaly detection:** z-score, Isolation Forest, Local Outlier Factor and One-Class SVM.
- **Classification:** Naive Bayes, decision tree, Random Forest, XGBoost, LightGBM, logistic regression, LSTM, attention and time-series transformer models.
- **RUL estimation:** CNN, LSTM, GRU and RNN hybrids used as feature extractors, with an XGBoost regressor on the extracted features.
- **Reactive power regression:** XGBoost, LightGBM, CatBoost and HistGradientBoosting.

## Results

- Compared classical machine-learning, deep-learning and hybrid models on three tasks: turbine status and event classification, Remaining Useful Life (RUL) estimation and reactive power regression. The metrics are in each notebook's saved outputs.
- Combined seven feature-selection methods by majority vote to reduce the input to 30 columns, mainly temperature, current, wind-speed and rotor/generator-speed signals.
- On the tabular classification tasks, tree-based models (decision tree, XGBoost) were competitive with the LSTM models.
- Gradient-boosted tree models fit reactive power well on a chronological hold-out split.
- Five hybrid architectures were compared for RUL estimation. CNN-LSTM-GRU ranked first in the original run, which was scored in-sample.

## Requirements

- Python 3.x
- Jupyter Notebook
- Libraries: pandas, numpy, scikit-learn, tensorflow, keras, xgboost, lightgbm, catboost, matplotlib, seaborn, statsmodels, imbalanced-learn

```bash
pip install pandas numpy scikit-learn tensorflow keras xgboost lightgbm catboost matplotlib seaborn statsmodels imbalanced-learn
```
