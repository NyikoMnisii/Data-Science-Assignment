import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score

# Assuming you have a DataFrame called df with the following columns
# Features: temperature, vibration, pressure, runtime
# Target: days_until_failure

df = pd.read_csv("Question 1 datasets .csv")
df['vibration_runtime'] = df['Vibration'] * df['Runtime']
df['log_pressure'] = np.log1p(df['Pressure'])
X = df[['Temperature', 'Vibration', 'Pressure', 'Runtime', 'vibration_runtime', 'log_pressure']]
y = df['Days to Failure']

# Train-test split (80/20)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
gbr.fit(X_train, y_train)

y_pred = gbr.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")
cv = KFold(n_splits=5, shuffle=True, random_state=42)

cv_rmse = -cross_val_score(gbr, X_scaled, y, scoring='neg_root_mean_squared_error', cv=cv)
cv_r2 = cross_val_score(gbr, X_scaled, y, scoring='r2', cv=cv)

print(f"Average RMSE (5-fold): {cv_rmse.mean():.2f}")
print(f"Average R² (5-fold): {cv_r2.mean():.2f}")
