import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc

# Load the dataset
merged_data = pd.read_csv('updated_merged_final_rounded.csv')

# Data Preprocessing
# Filter the dataset based on the criteria in the flowchart (age >= 18 and ICU stay >= 72 hours)
filtered_data = merged_data[(merged_data['age'] >= 18) & (merged_data['icu_stay_hours'] >= 72)]

# Feature selection
features = ['age', 'icu_stay_hours']  # Removed 'ventilation_status_flag' from features
X = filtered_data[features]
y = filtered_data['ventilation_status_flag']  # Assuming this is the target variable

# Address class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# KNN Model with Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
knn_accuracies = []
knn_auc_scores = []

for train_index, val_index in skf.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_fold, y_train_fold)
    y_val_pred = knn.predict(X_val_fold)
    knn_accuracies.append(accuracy_score(y_val_fold, y_val_pred))
    fpr, tpr, _ = roc_curve(y_val_fold, knn.predict_proba(X_val_fold)[:, 1])
    knn_auc_scores.append(auc(fpr, tpr))

print("KNN Cross-Validation Accuracy:", np.mean(knn_accuracies))
print("KNN Cross-Validation AUC-ROC:", np.mean(knn_auc_scores))

# Final KNN Model
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
y_pred_proba = knn.predict_proba(X_test)[:, 1]
knn_auc = roc_auc_score(y_test, y_pred_proba)
print("KNN Model Test Accuracy:", accuracy_score(y_test, y_pred))
print("KNN Model Test AUC-ROC:", knn_auc)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Model Performance Summary
print("\nModel Performance Summary:")
print("KNN Model:")
print(f"AUC-ROC: {knn_auc:.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {classification_report(y_test, y_pred, output_dict=True)['weighted avg']['precision']:.4f}")
print(f"Recall: {classification_report(y_test, y_pred, output_dict=True)['weighted avg']['recall']:.4f}")
print(f"F1 Score: {classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']:.4f}")

