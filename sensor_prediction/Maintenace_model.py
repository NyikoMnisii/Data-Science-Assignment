import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("Question 1 datasets .csv")


def remove_outliers_iqr(data, columns):
    df_clean = data.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    return df_clean

sensor_columns = ['Temperature', 'Vibration', 'Pressure', 'Runtime', 'Days to Failure']
df_clean = remove_outliers_iqr(df, sensor_columns)

print(f"Original rows: {df.shape[0]}, After outlier removal: {df_clean.shape[0]}")

# Feature and target selection
features = ['Temperature', 'Vibration', 'Pressure', 'Runtime']
X = df_clean[features]
y = df_clean['Days to Failure']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


def augment_with_noise(X, y, noise_level=0.03, num_augments=3):
    X_aug = [X]
    y_aug = [y]
    for _ in range(num_augments):
        noise = np.random.normal(0, noise_level, X.shape)
        X_noisy = X + noise
        X_aug.append(X_noisy)
        y_aug.append(y)  # Keep original target
    return np.vstack(X_aug), np.hstack(y_aug)

X_aug, y_aug = augment_with_noise(X_scaled, y.to_numpy())
print(f"After augmentation: {X_aug.shape[0]} samples")


kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
clusters = kmeans.fit_predict(X_aug)


X_clustered = np.column_stack((X_aug, clusters))


X_train, X_test, y_train, y_test = train_test_split(X_clustered, y_aug, test_size=0.2, random_state=42)

# Train model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)


y_pred = rf.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)


kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmse = -cross_val_score(rf, X_clustered, y_aug, scoring='neg_root_mean_squared_error', cv=kf)
cv_r2 = cross_val_score(rf, X_clustered, y_aug, scoring='r2', cv=kf)

# Final output
print(f"\n With Outlier Removal + Gaussian Augmentation")
print(f"Cross-Validated RMSE: {cv_rmse.mean():.2f}")
print(f"Cross-Validated R²: {cv_r2.mean():.2f}")
print(f"Test RMSE: {rmse:.2f}")
print(f"Test R²: {r2:.2f}")
