import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE


df = pd.read_csv("Question 2 Datasets .csv")
df.columns = df.columns.str.strip()

# Ensure Is_Fraud is integer and consistent
df['Is_Fraud'] = df['Is_Fraud'].astype(int)


df_labeled = df[df['Is_Fraud'] != -1].copy()
df_unlabeled = df[df['Is_Fraud'] == -1].copy()


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

# Augmentation of minority labeled data with noise
numeric_cols = ['Amount', 'Time_Hour']
cat_cols = ['Location', 'Merchant', 'Time_Period']


def augment_minority_only(df, numeric_columns, label_column, minority_class=1, noise_level=0.05, num_augments=9):
    df_minority = df[df[label_column] == minority_class]
    df_majority = df[df[label_column] != minority_class]

    X_num = df_minority[numeric_columns].values
    y = df_minority[label_column].values

    X_aug = [X_num]
    y_aug = [y]

    for _ in range(num_augments):
        noise = np.random.normal(0, noise_level * np.std(X_num, axis=0), X_num.shape)
        X_noisy = X_num + noise
        X_aug.append(X_noisy)
        y_aug.append(y)

    X_augmented = np.vstack(X_aug)
    y_augmented = np.hstack(y_aug)

    df_cat_repeated = pd.concat([df_minority[cat_cols]] * (num_augments + 1), ignore_index=True)

    df_minority_aug = pd.DataFrame(X_augmented, columns=numeric_columns)
    df_minority_aug[label_column] = y_augmented
    df_minority_aug = pd.concat([df_minority_aug, df_cat_repeated], axis=1)


    df_combined = pd.concat([df_majority, df_minority_aug], ignore_index=True)

    return df_combined


df_augmented = augment_minority_only(df_labeled, numeric_cols, 'Is_Fraud', noise_level=0.1, num_augments=14)

print(f"Original labeled data shape: {df_labeled.shape}")
print(f"Augmented labeled data shape: {df_augmented.shape}")
print("Class distribution after augmentation:")
print(df_augmented['Is_Fraud'].value_counts())



cluster_features = ['Amount', 'Time_Hour', 'Location', 'Merchant', 'Time_Period']
X_cluster = df_unlabeled[cluster_features]


cluster_preprocessor = ColumnTransformer([
    ("onehot", OneHotEncoder(drop='first'), ['Location', 'Merchant', 'Time_Period']),
    ("scale", StandardScaler(), ['Amount', 'Time_Hour'])
])

X_cluster_preprocessed = cluster_preprocessor.fit_transform(X_cluster)


if hasattr(X_cluster_preprocessed, "toarray"):
    X_cluster_preprocessed = X_cluster_preprocessed.toarray()

# Elbow method to select number of clusters k
sse = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X_cluster_preprocessed)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 10), sse, marker='o')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of clusters")
plt.ylabel("SSE")
plt.grid(True)
plt.show()


kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
df_unlabeled['Cluster'] = kmeans.fit_predict(X_cluster_preprocessed)

print("Clusters assigned to unlabeled data:")
print(df_unlabeled['Cluster'].value_counts())



features = ['Amount', 'Time_Hour', 'Location', 'Merchant', 'Time_Period']
X = df_augmented[features]
y = df_augmented['Is_Fraud']

#
preprocessor = ColumnTransformer([
    ("onehot", OneHotEncoder(drop='first'), ['Location', 'Merchant', 'Time_Period']),
    ("scale", StandardScaler(), ['Amount', 'Time_Hour'])
])

smote = SMOTE(random_state=42)


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

f1_scores = []

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]


    X_train_proc = preprocessor.fit_transform(X_train)
    if hasattr(X_train_proc, "toarray"):
        X_train_proc = X_train_proc.toarray()

    X_test_proc = preprocessor.transform(X_test)
    if hasattr(X_test_proc, "toarray"):
        X_test_proc = X_test_proc.toarray()

    # Apply SMOTE only on training data
    X_train_res, y_train_res = smote.fit_resample(X_train_proc, y_train)


    clf = GaussianNB()
    clf.fit(X_train_res, y_train_res)


    y_pred = clf.predict(X_test_proc)


    f1 = f1_score(y_test, y_pred)
    f1_scores.append(f1)

print(f"Average F1 Score (10-fold CV): {np.mean(f1_scores):.3f}")



X_full_proc = preprocessor.fit_transform(X)
if hasattr(X_full_proc, "toarray"):
    X_full_proc = X_full_proc.toarray()

X_resampled, y_resampled = smote.fit_resample(X_full_proc, y)

final_model = GaussianNB()
final_model.fit(X_resampled, y_resampled)


ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix (Last Fold)")
plt.show()
