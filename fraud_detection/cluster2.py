import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("Question 2 Datasets .csv")
df.columns = df.columns.str.strip()

# Separate unlabeled data
df_unlabeled = df[df['Is_Fraud'] == -1].copy()

# Bin Time into Periods
def time_bin(hour):
    if 0 <= hour < 6:
        return 'Night'
    elif 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    else:
        return 'Evening'

df_unlabeled['Time_Period'] = df_unlabeled['Time_Hour'].apply(time_bin)

# Preprocessing for anomaly detection
features = ['Amount', 'Time_Hour', 'Location', 'Merchant', 'Time_Period']
X_unlabeled = df_unlabeled[features].copy()

preprocessor = ColumnTransformer([
    ("onehot", OneHotEncoder(drop='first'), ['Location', 'Merchant', 'Time_Period']),
    ("scale", StandardScaler(), ['Amount', 'Time_Hour'])
])

X_processed = preprocessor.fit_transform(X_unlabeled)

# Fit Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
anomaly_labels = iso_forest.fit_predict(X_processed)

# Add predictions to DataFrame
df_unlabeled['Anomaly'] = anomaly_labels  # -1 = anomaly, 1 = normal

# Show result counts
print("Anomaly Counts:\n", df_unlabeled['Anomaly'].value_counts())

# Show top suspicious transactions
print("\nTop Suspicious Transactions:\n", df_unlabeled[df_unlabeled['Anomaly'] == -1].sort_values(by='Amount', ascending=False).head(10)[['Index', 'Amount', 'Time_Hour', 'Location', 'Merchant', 'Anomaly']])
