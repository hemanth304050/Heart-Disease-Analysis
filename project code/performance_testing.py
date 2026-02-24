"""
Performance Testing - Train ML models and evaluate heart disease prediction.
Saves model + metrics used by the Flask dashboard.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle, json, os

os.makedirs("static/charts", exist_ok=True)
os.makedirs("data", exist_ok=True)

df = pd.read_csv("data/heart_clean.csv")

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
X = df[FEATURES]
y = df["target"]

scaler    = StandardScaler()
X_scaled  = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y)

# ── Train 4 models ────────────────────────────────────────────────────────────
models = {
    "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":        RandomForestClassifier(n_estimators=150, random_state=42),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=150, random_state=42),
    "SVM":                  SVC(probability=True, random_state=42),
}

results    = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("=" * 60)
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)[:, 1]
    acc     = accuracy_score(y_test, y_pred) * 100
    auc     = roc_auc_score(y_test, y_prob) * 100
    cv_acc  = cross_val_score(model, X_scaled, y, cv=cv, scoring="accuracy").mean() * 100
    report  = classification_report(y_test, y_pred, output_dict=True)
    results[name] = {
        "accuracy": round(acc, 2),
        "auc":      round(auc, 2),
        "cv_acc":   round(cv_acc, 2),
        "precision": round(report["weighted avg"]["precision"] * 100, 2),
        "recall":    round(report["weighted avg"]["recall"] * 100, 2),
        "f1":        round(report["weighted avg"]["f1-score"] * 100, 2),
        "cm":        confusion_matrix(y_test, y_pred).tolist(),
    }
    print(f"{name:<26}  Acc={acc:.1f}%  AUC={auc:.1f}%  CV-Acc={cv_acc:.1f}%")

print("=" * 60)

# ── Pick best model (by AUC) ─────────────────────────────────────────────────
best_name  = max(results, key=lambda k: results[k]["auc"])
best_model = models[best_name]
print(f"\nBest model : {best_name}")

with open("data/model.pkl",  "wb") as f: pickle.dump(best_model, f)
with open("data/scaler.pkl", "wb") as f: pickle.dump(scaler, f)
with open("data/features.pkl","wb") as f: pickle.dump(FEATURES, f)
with open("data/metrics.json","w")  as f: json.dump(results, f, indent=2)
print("[✓] Model, scaler, features and metrics saved.")

# ── Chart 9: Model Accuracy Comparison ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
names = list(results.keys())
accs  = [results[m]["accuracy"] for m in names]
aucs  = [results[m]["auc"]      for m in names]
x = np.arange(len(names)); w = 0.35
bars1 = ax.bar(x - w/2, accs, w, label="Accuracy (%)", color="#2196F3", edgecolor="white")
bars2 = ax.bar(x + w/2, aucs, w, label="AUC (%)",      color="#4CAF50", edgecolor="white")
for b in bars1 + bars2:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
            f"{b.get_height():.1f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
ax.set_ylim(50, 105); ax.set_ylabel("Score (%)"); ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
ax.legend(); plt.tight_layout()
plt.savefig("static/charts/09_model_comparison.png", dpi=110, bbox_inches="tight")
plt.close(); print("[✓] Saved static/charts/09_model_comparison.png")

# ── Chart 10: Feature Importance (Random Forest) ────────────────────────────
rf = models["Random Forest"]
fi = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(7, 5))
fi.plot(kind="barh", ax=ax, color="#FF9800", edgecolor="white")
ax.set_title("Feature Importance (Random Forest)", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("static/charts/10_feature_importance.png", dpi=110, bbox_inches="tight")
plt.close(); print("[✓] Saved static/charts/10_feature_importance.png")

# ── Chart 11: ROC Curves ─────────────────────────────────────────────────────
colors = ["#2196F3","#4CAF50","#FF9800","#9C27B0"]
fig, ax = plt.subplots(figsize=(7, 5))
for (name, model), col in zip(models.items(), colors):
    prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    ax.plot(fpr, tpr, color=col, label=f"{name} (AUC={results[name]['auc']:.1f}%)")
ax.plot([0,1],[0,1],"k--",linewidth=1)
ax.set_title("ROC Curves - All Models", fontsize=13, fontweight="bold")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.legend(fontsize=8); plt.tight_layout()
plt.savefig("static/charts/11_roc_curves.png", dpi=110, bbox_inches="tight")
plt.close(); print("[✓] Saved static/charts/11_roc_curves.png")

print(f"\n[✓] Performance testing complete. Best: {best_name}")
