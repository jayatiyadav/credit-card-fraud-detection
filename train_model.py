import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from imblearn.over_sampling import SMOTE


# ==========================================
# 1. LOAD DATA
# ==========================================

print("Loading dataset...")

df = pd.read_csv("data/creditcard.csv")

print("Original dataset shape:", df.shape)


# ==========================================
# 2. REMOVE DUPLICATES
# ==========================================

duplicates = df.duplicated().sum()

print("Duplicate rows found:", duplicates)

df = df.drop_duplicates()

print("Dataset after removing duplicates:", df.shape)


# ==========================================
# 3. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("Class", axis=1)
y = df["Class"]

print("\nClass distribution:")
print(y.value_counts())


# ==========================================
# 4. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ==========================================
# 5. FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==========================================
# 6. HANDLE CLASS IMBALANCE USING SMOTE
# ==========================================

print("\nApplying SMOTE...")

smote = SMOTE(random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train,
    y_train
)

print("Before SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(pd.Series(y_train_resampled).value_counts())


# ==========================================
# 7. LOGISTIC REGRESSION
# ==========================================

print("\n" + "=" * 50)
print("LOGISTIC REGRESSION")
print("=" * 50)

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(
    X_train_resampled,
    y_train_resampled
)

logistic_predictions = logistic_model.predict(X_test)
logistic_probabilities = logistic_model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        logistic_predictions,
        target_names=["Legitimate", "Fraud"]
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, logistic_predictions))

logistic_auc = roc_auc_score(
    y_test,
    logistic_probabilities
)

print("ROC-AUC Score:", logistic_auc)


# ==========================================
# 8. RANDOM FOREST
# ==========================================

print("\n" + "=" * 50)
print("RANDOM FOREST")
print("=" * 50)

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

random_forest.fit(
    X_train_resampled,
    y_train_resampled
)

rf_predictions = random_forest.predict(X_test)
rf_probabilities = random_forest.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        rf_predictions,
        target_names=["Legitimate", "Fraud"]
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))

rf_auc = roc_auc_score(
    y_test,
    rf_probabilities
)

print("ROC-AUC Score:", rf_auc)


# ==========================================
# 9. SELECT BEST MODEL
# ==========================================

if rf_auc > logistic_auc:
    best_model = random_forest
    best_model_name = "Random Forest"
    best_auc = rf_auc
else:
    best_model = logistic_model
    best_model_name = "Logistic Regression"
    best_auc = logistic_auc


print("\n" + "=" * 50)
print("BEST MODEL")
print("=" * 50)

print("Model:", best_model_name)
print("ROC-AUC:", best_auc)


# ==========================================
# 10. SAVE MODEL + SCALER
# ==========================================

joblib.dump(best_model, "models/fraud_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nModel saved to: models/fraud_model.pkl")
print("Scaler saved to: models/scaler.pkl")

print("\nTraining completed successfully! 🚀")