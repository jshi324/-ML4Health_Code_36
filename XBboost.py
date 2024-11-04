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



# XGBoost Model with Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
xgb_accuracies = []
xgb_auc_scores = []

for train_index, val_index in skf.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    xgb_model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, learning_rate=0.05, max_depth=4,
                                  use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train_fold, y_train_fold)
    y_val_pred_xgb = xgb_model.predict(X_val_fold)
    xgb_accuracies.append(accuracy_score(y_val_fold, y_val_pred_xgb))
    fpr, tpr, _ = roc_curve(y_val_fold, xgb_model.predict_proba(X_val_fold)[:, 1])
    xgb_auc_scores.append(auc(fpr, tpr))

print("XGBoost Cross-Validation Accuracy:", np.mean(xgb_accuracies))
print("XGBoost Cross-Validation AUC-ROC:", np.mean(xgb_auc_scores))

# Final XGBoost Model
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
xgb_auc = roc_auc_score(y_test, y_pred_proba_xgb)
print("XGBoost Model Test Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("XGBoost Model Test AUC-ROC:", xgb_auc)
print("Classification Report:")
print(classification_report(y_test, y_pred_xgb))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))

# Model Performance Summary
print("\nModel Performance Summary:")

print("\nXGBoost Model:")
print(f"AUC-ROC: {xgb_auc:.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print(f"Precision: {classification_report(y_test, y_pred_xgb, output_dict=True)['weighted avg']['precision']:.4f}")
print(f"Recall: {classification_report(y_test, y_pred_xgb, output_dict=True)['weighted avg']['recall']:.4f}")
print(f"F1 Score: {classification_report(y_test, y_pred_xgb, output_dict=True)['weighted avg']['f1-score']:.4f}")