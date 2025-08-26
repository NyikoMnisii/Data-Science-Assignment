import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE

# Load CSV
df = pd.read_csv("Question 2 Datasets .csv")
df.columns = df.columns.str.strip()  # Strip whitespace from headers
df['Is_Fraud'] = df['Is_Fraud'].astype(int)

# Split labeled/unlabeled
df_labeled = df[df['Is_Fraud'] != -1].copy()
df_unlabeled = df[df['Is_Fraud'] == -1].copy()

# --- Feature Engineering ---
def time_bin(hour):
    if 0 <= hour < 6:
        return 'Night'
    elif 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    else:
        return 'Evening'

df_labeled['Time_Period'] = df_labeled['Time_Hour'].apply(time_bin)
df_unlabeled['Time_Period'] = df_unlabeled['Time_Hour'].apply(time_bin)

# --- Clustering (Unsupervised) ---
cluster_features = ['Amount', 'Time_Hour', 'Location', 'Merchant', 'Time_Period']
X_cluster = df_unlabeled[cluster_features]

cluster_preprocessor = ColumnTransformer([
    ("onehot", OneHotEncoder(drop='first'), ['Location', 'Merchant', 'Time_Period']),
    ("scale", StandardScaler(), ['Amount', 'Time_Hour'])
])

X_cluster_preprocessed = cluster_preprocessor.fit_transform(X_cluster)

# Elbow Method
sse = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X_cluster_preprocessed)
    sse.append(kmeans.inertia_)

plt.plot(range(1, 10), sse, marker='o')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of clusters")
plt.ylabel("SSE")
plt.grid()
plt.show()

# Final Clustering with k=3
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
df_unlabeled['Cluster'] = kmeans.fit_predict(X_cluster_preprocessed)

# --- Classification (Supervised) ---

# Features and label
features = ['Amount', 'Time_Hour', 'Location', 'Merchant', 'Time_Period']
X = df_labeled[features]
y = df_labeled['Is_Fraud'].astype(int)

# Preprocessing pipeline (same as clustering)
preprocessor = ColumnTransformer([
    ("onehot", OneHotEncoder(drop='first'), ['Location', 'Merchant', 'Time_Period']),
    ("scale", StandardScaler(), ['Amount', 'Time_Hour'])
])

# Train-test split BEFORE SMOTE
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Apply preprocessing
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Only apply SMOTE if safe
if df_labeled['Is_Fraud'].value_counts()[1] >= 6:
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_processed, y)

# Apply SMOTE to training data
smote = SMOTE(random_state=42, k_neighbors=3)
X_resampled, y_resampled = smote.fit_resample(X_train_processed, y_train)

# Train classifier
model = GaussianNB()
model.fit(X_resampled, y_resampled)

# Predict
y_pred = model.predict(X_test_processed)

# Evaluation
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("F1 Score:", round(f1, 3))
print("Confusion Matrix:\n", cm)

# Display confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()

# Cross-validation with preprocessing in pipeline
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GaussianNB())
])

cv_scores = cross_val_score(full_pipeline, X, y, scoring='f1', cv=10)
print("Average F1 Score (10-fold):", round(cv_scores.mean(), 3))
print(df['Is_Fraud'].value_counts())

