#  Predictive Maintenance in Manufacturing


##  Objective

Build a predictive model to estimate the number of **days until machine failure** using industrial sensor data, including:

- **Temperature (°C)**
- **Vibration (mm/s)**
- **Pressure (psi)**
- **Runtime (hours)**

The target is a continuous value: **Days to Failure**.

---

##  Problem Statement

Initial attempts with linear regression overfitted the training data, performing poorly on unseen sensor readings. This project applies an **ensemble learning approach (Random Forest)** and integrates **clustering, feature scaling, and data augmentation** to improve generalization.

---

##  Steps Taken

### 1.  Data Exploration
- Loaded the dataset from `Question 1 datasets .csv`.
- Explored feature types and checked for missing values.

### 2.  Outlier Removal
- Removed outliers using the **IQR method** for columns:
  - `Temperature`, `Vibration`, `Pressure`, `Runtime`, `Days to Failure`
- No outliers were found, but method remains for robustness.

### 3.  Feature Engineering
- **Features selected**:
  - `Temperature`, `Vibration`, `Pressure`, `Runtime`
- **Target**: `Days to Failure`

### 4.  Feature Scaling
- Standardized features using **StandardScaler** for consistent scale, especially useful for clustering.

### 5.  Data Augmentation
- Created synthetic data using **Gaussian noise** on sensor values.
- Increased dataset size from **200 → 800 samples**.
- Preserved original target values to retain true failure patterns.

### 6.  Unsupervised Learning (Clustering)
- Applied **K-Means (k=5)** to identify hidden patterns in the data.
- Added **cluster labels** as an extra feature to assist the regression model.

### 7.  Model Training – Random Forest
- Trained **RandomForestRegressor** with:
  - `n_estimators=100`
  - `random_state=42`
- Used an **80/20 train-test split**.

### 8.  Evaluation
- Metrics used:
  - **RMSE (Root Mean Squared Error)**
  - **R² (Coefficient of Determination)**
- Also performed **5-fold cross-validation** to evaluate model generalization.

---

##  Results

| Metric                  | Value      |
|--------------------------|------------|
| Test RMSE               | **47.69**   |
| Test R²                 | **0.87**    |
| Cross-Validated RMSE    | **47.52**   |
| Cross-Validated R²      | **0.87**    |

 RMSE < 50 indicates good accuracy  
 R² ≈ 0.87 means strong generalization and fit  

---

##  Why Random Forest?

- **Reduces overfitting** by averaging results from multiple trees (bagging)
- Handles non-linear relationships well
- Naturally robust to noise and feature scaling

---

# Fraud Detection in Banking  

##  Objective

This project focuses on detecting fraudulent banking transactions using both **unsupervised (clustering)** and **supervised (classification)** learning techniques. The dataset includes both labeled and unlabeled transactions, and the goal is to identify patterns of fraud through clustering and classify future transactions using a robust classification pipeline.

---

## Dataset Overview

The dataset includes the following features:

- `Amount`: Transaction amount (USD)
- `Time_Hour`: Hour of the day (0–23)
- `Location`: Transaction location category (e.g., ATM, Online)
- `Merchant`: Type of merchant
- `Is_Fraud`: Label (`0` = Not fraud, `1` = Fraud, `-1` = Unlabeled)

---

##  Problem Breakdown

###  A. Clustering Unlabeled Data

- **Algorithm:** K-Means (unsupervised)
- **Justification:** Allows grouping of transactions to find potential anomaly clusters (e.g., high-amount, late-night transactions).
- **Feature Engineering:**
  - One-Hot Encoding for categorical features (`Location`, `Merchant`, `Time_Period`)
  - StandardScaler for numeric features (`Amount`, `Time_Hour`)
  - `Time_Period` feature engineered by binning `Time_Hour` into: Morning, Afternoon, Evening, Night
- **Elbow Method:** Used to choose optimal number of clusters (`k=3`)
- **Result:** Labeled clusters for further inspection and analysis

```text

Clusters assigned to unlabeled data:
Cluster 1: 40 samples
Cluster 2: 33 samples
Cluster 0: 27 samples
```
### Feature Engineering & Class Balancing:

- Binned `Time_Hour` into time periods for better feature representation.
- Applied One-Hot Encoding to categorical features.
- Scaled numerical features.
- Augmented minority class (fraud cases) using Gaussian noise-based synthetic samples to reduce class imbalance.
- Further applied SMOTE during training to balance classes dynamically in cross-validation folds.

---

### Model:

- Gaussian Naive Bayes was chosen for handling categorical data efficiently with low variance.
- Evaluated using 10-fold Stratified Cross-Validation.

---

### Performance:

- Average F1 Score (10-fold CV): 0.822







