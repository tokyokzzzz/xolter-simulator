import pickle

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize


# ── 1. Load dataset ──────────────────────────────────────────────────────────

print("Loading dataset...")
df = pd.read_csv("data/training_dataset.csv")

X = df.drop(columns=["label"]).values
y = df["label"].values

print(f"  Samples  : {len(df)}")
print(f"  Features : {X.shape[1]}")
print(f"  Classes  : {sorted(set(y))}")


# ── 2. Scale features ────────────────────────────────────────────────────────

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

with open("data/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("Scaler saved -> data/scaler.pkl")


# ── 3. Encode labels ─────────────────────────────────────────────────────────

le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = list(le.classes_)

with open("data/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
print(f"Label encoder saved -> data/label_encoder.pkl  classes={class_names}")


# ── 4. Train / test split ────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\nTrain samples : {len(X_train)}  |  Test samples : {len(X_test)}")


# ── 5. Define models ─────────────────────────────────────────────────────────

mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation="relu",
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42,
)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

ensemble = VotingClassifier(
    estimators=[("mlp", mlp), ("rf", rf)],
    voting="soft",
)


# ── 6. Train ─────────────────────────────────────────────────────────────────

print("\nTraining ensemble model (MLP + Random Forest)...")
ensemble.fit(X_train, y_train)
print("Training complete.")


# ── 7. Evaluate ──────────────────────────────────────────────────────────────

y_pred = ensemble.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(y_test, y_pred, target_names=class_names))


# ── 8. Confusion matrix ──────────────────────────────────────────────────────

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    ax=ax,
)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_title("Holter Monitor — Confusion Matrix", fontsize=14)
plt.tight_layout()
plt.savefig("data/confusion_matrix.png", dpi=150)
plt.close()
print("Confusion matrix saved -> data/confusion_matrix.png")


# ── 9. ROC curves ────────────────────────────────────────────────────────────

n_classes = len(class_names)
y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
y_prob = ensemble.predict_proba(X_test)

colors = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4"]
fig, ax = plt.subplots(figsize=(10, 7))

for i, (name, color) in enumerate(zip(class_names, colors)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("Holter Monitor — ROC Curves (One-vs-Rest)", fontsize=14)
ax.legend(loc="lower right", fontsize=10)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig("data/roc_curves.png", dpi=150)
plt.close()
print("ROC curves saved -> data/roc_curves.png")


# ── 10. Save model ───────────────────────────────────────────────────────────

with open("data/holter_model.pkl", "wb") as f:
    pickle.dump(ensemble, f)

print("\nModel saved -> data/holter_model.pkl")
print("=" * 60)
print("Model saved. Ready for deployment.")
print(f"Overall Accuracy: {accuracy * 100:.1f}%")
print("=" * 60)


if __name__ == "__main__":
    pass  # all steps execute at module level when run directly
