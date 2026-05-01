"""
train_model.py
--------------
Phase 3 — Model Training

What this does:
  - Loads dataset/landmarks.csv
  - Splits into train/test sets (80/20)
  - Trains a Random Forest classifier
  - Evaluates accuracy + confusion matrix
  - Saves trained model to models/random_forest.pkl

Run: python src\train_model.py
"""

import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ── Load dataset ──────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("dataset/landmarks.csv")

print(f"  Total samples : {len(df)}")
print(f"  Features      : {df.shape[1] - 1}")
print(f"  Classes       : {df['label'].unique()}")

# ── Split features and labels ─────────────────────────────────────────────────
# X = all columns except label (the 63 landmark coordinates)
# y = just the label column
X = df.drop("label", axis=1).values   # shape: (1000, 63)
y = df["label"].values                 # shape: (1000,)

# ── Train/test split ──────────────────────────────────────────────────────────
# test_size=0.2  → 20% for testing (200 samples), 80% for training (800 samples)
# random_state=42 → fixes the random split so results are reproducible
# stratify=y → ensures each class has equal representation in both splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTrain samples : {len(X_train)}")
print(f"Test samples  : {len(X_test)}")

# ── Train Random Forest ───────────────────────────────────────────────────────
# n_estimators=100  → 100 decision trees in the forest
# random_state=42   → reproducible results
print("\nTraining Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1        # use all CPU cores → faster training
)
model.fit(X_train, y_train)
print("  Training complete!")

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*50}")
print(f"  Test Accuracy: {accuracy * 100:.2f}%")
print(f"{'='*50}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ── Confusion Matrix ──────────────────────────────────────────────────────────
# Rows = actual class, Columns = predicted class
# Perfect model = diagonal is all high numbers, rest are 0
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title(f"Confusion Matrix — Accuracy: {accuracy*100:.2f}%")
plt.tight_layout()

os.makedirs("models", exist_ok=True)
plt.savefig("models/confusion_matrix.png", dpi=150)
print("\nConfusion matrix saved to models/confusion_matrix.png")
plt.show()

# ── Feature Importance ────────────────────────────────────────────────────────
# Which of the 63 landmark coordinates mattered most?
importances = model.feature_importances_
feature_names = df.drop("label", axis=1).columns

top_indices = np.argsort(importances)[::-1][:15]  # top 15 features
top_names   = feature_names[top_indices]
top_values  = importances[top_indices]

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.bar(range(15), top_values, color="steelblue")
ax2.set_xticks(range(15))
ax2.set_xticklabels(top_names, rotation=45, ha="right")
ax2.set_title("Top 15 Most Important Features")
ax2.set_ylabel("Importance Score")
plt.tight_layout()
plt.savefig("models/feature_importance.png", dpi=150)
print("Feature importance saved to models/feature_importance.png")
plt.show()

# ── Save model ────────────────────────────────────────────────────────────────
model_path = "models/random_forest.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"\nModel saved to {model_path}")
print("\nDone! Ready for Phase 4 — Live Prediction.")