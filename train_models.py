"""
NeuroNetics-SmartEdu - ML Model Training
Trains regression (final score prediction) and classifier (at-risk detection)
"""
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, mean_squared_error

# ── Synthetic training data ──────────────────────────────────────────────────
np.random.seed(42)
N = 500

quiz        = np.random.randint(30, 100, N)
attendance  = np.random.randint(40, 100, N)
study_time  = np.random.randint(1,  10,  N)

# Final score formula with some noise
final = (
    0.45 * quiz +
    0.30 * attendance +
    0.25 * study_time * 5 +
    np.random.normal(0, 5, N)
).clip(0, 100)

# At-risk: final < 60 OR attendance < 60
risk = ((final < 60) | (attendance < 60)).astype(int)

X = np.column_stack([quiz, attendance, study_time])
y_reg = final
y_clf = risk

X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=42
)

# ── Scaler ───────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── Regression model ─────────────────────────────────────────────────────────
reg = GradientBoostingRegressor(n_estimators=150, max_depth=4, random_state=42)
reg.fit(X_train_s, yr_train)
rmse = np.sqrt(mean_squared_error(yr_test, reg.predict(X_test_s)))
print(f"[Regressor] RMSE: {rmse:.2f}")

# ── Classifier model ─────────────────────────────────────────────────────────
clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
clf.fit(X_train_s, yc_train)
print("[Classifier] Report:")
print(classification_report(yc_test, clf.predict(X_test_s), target_names=["Safe","At-Risk"]))

# ── Save ─────────────────────────────────────────────────────────────────────
os.makedirs("saved_models", exist_ok=True)
joblib.dump(reg,    "saved_models/regressor.pkl")
joblib.dump(clf,    "saved_models/classifier.pkl")
joblib.dump(scaler, "saved_models/scaler.pkl")
print("Models saved to ml/saved_models/")