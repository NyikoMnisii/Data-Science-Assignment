# Data-Science-Assignment

# Problem 1 - Predictive Maintenance in Manufacturing (Regression with Overfitting Resolution) 

## Objective
To build a regression model that predicts **Days Until Failure** for heavy machinery using industrial sensor data. The dataset includes:

- Temperature (°C)
- Vibration (mm/s)
- Pressure (psi)
- Runtime (hours)

Target: **Days to Failure** (continuous)

---

##  Problem Statement
Initial linear regression models **overfit** the training data and performed poorly on unseen samples. The goal is to apply **ensemble learning** to improve generalization and reduce overfitting.

---

##  Approach

###  1. Data Preprocessing
- Dropped irrelevant columns (`Index`)
- Checked for missing values (none found)
- Explored data distributions

### 2. Feature Engineering
- Created interaction feature: `vibration_runtime = Vibration × Runtime`
- Applied log transform: `log_pressure = log(Pressure + 1)`
- Scaled features using `StandardScaler`

###  3. Models Used
- **Linear Regression** (baseline)
- **Random Forest Regressor**
- **Gradient Boosting Regressor**

###  4. Evaluation Metrics
- **RMSE** (Root Mean Squared Error)
- **R² Score**
- **5-Fold Cross-Validation** for generalization

---

##  Results

| Model                  | RMSE    | R² Score | CV RMSE | CV R² |
|------------------------|---------|----------|---------|-------|
| Linear Regression      | High    | Negative | High    | Negative |
| Random Forest Regressor| 159.22  | -0.20    | 141.67  | -0.16 |
| Gradient Boosting      | 173.08  | -0.42    | 154.56  | -0.39 |

> **Note**: Despite ensemble methods, performance remained poor due to potential low signal or noise in the dataset (possibly intentional for academic purposes).

---

## conclusion

- **Ensemble models reduce overfitting**, but cannot fix weak data.
- **Feature engineering** helped marginally, but more informative features are needed.
- Dataset likely requires **domain-specific attributes** (e.g., machine ID, timestamps, failure type).

---

##  Problem 2

Due to extreme class imbalance (only 5 labeled fraud cases), classification performance is low. F1 = 0.16. To improve performance, more labeled fraud examples are needed or an anomaly detection approach should be considered."

