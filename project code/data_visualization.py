"""
Data Visualization - Heart Disease Analysis
Generates and saves 8 charts as PNG files in static/charts/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("static/charts", exist_ok=True)
df = pd.read_csv("data/heart_clean.csv")

PALETTE  = ["#2196F3", "#F44336"]
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("ggplot")

def save(name):
    plt.tight_layout()
    plt.savefig(f"static/charts/{name}.png", dpi=110, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved static/charts/{name}.png")

# ── 1. Target Distribution ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
counts  = df["target"].value_counts()
bars    = ax.bar(["No Disease", "Disease"], counts.values, color=PALETTE, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            f"{val}\n({val/len(df)*100:.1f}%)", ha="center", fontsize=10, fontweight="bold")
ax.set_title("Heart Disease Distribution", fontsize=14, fontweight="bold")
ax.set_ylabel("Count"); ax.set_ylim(0, counts.max() * 1.2)
save("01_target_distribution")

# ── 2. Age Distribution by Target ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
for t, lbl, c in zip([0, 1], ["No Disease", "Disease"], PALETTE):
    ax.hist(df[df["target"] == t]["age"], bins=20, alpha=0.7, label=lbl, color=c, edgecolor="white")
ax.set_title("Age Distribution by Heart Disease Status", fontsize=14, fontweight="bold")
ax.set_xlabel("Age"); ax.set_ylabel("Count"); ax.legend()
save("02_age_distribution")

# ── 3. Correlation Heatmap ───────────────────────────────────────────────────
num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca", "target"]
fig, ax  = plt.subplots(figsize=(8, 6))
corr     = df[num_cols].corr()
mask     = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, linewidths=0.5, ax=ax, annot_kws={"size": 9})
ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold")
save("03_correlation_heatmap")

# ── 4. Chest Pain Type vs Target ─────────────────────────────────────────────
cp_labels = {0: "Typical\nAngina", 1: "Atypical\nAngina", 2: "Non-Anginal", 3: "Asymptomatic"}
df["cp_label"] = df["cp"].map(cp_labels)
ct = pd.crosstab(df["cp_label"], df["target"])
ct.columns = ["No Disease", "Disease"]
fig, ax = plt.subplots(figsize=(8, 4))
ct.plot(kind="bar", ax=ax, color=PALETTE, edgecolor="white")
ax.set_title("Chest Pain Type vs Heart Disease", fontsize=14, fontweight="bold")
ax.set_xlabel("Chest Pain Type"); ax.set_ylabel("Count")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(title="Status")
save("04_chest_pain_vs_target")

# ── 5. Cholesterol by Age Group ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
for t, lbl, c in zip([0, 1], ["No Disease", "Disease"], PALETTE):
    sub = df[df["target"] == t]
    ax.scatter(sub["age"], sub["chol"], alpha=0.5, s=20, label=lbl, color=c)
ax.axhline(240, color="orange", linestyle="--", linewidth=1.5, label="High Chol (240)")
ax.set_title("Cholesterol vs Age", fontsize=14, fontweight="bold")
ax.set_xlabel("Age"); ax.set_ylabel("Cholesterol (mg/dl)"); ax.legend()
save("05_cholesterol_vs_age")

# ── 6. Max Heart Rate vs Age ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
for t, lbl, c in zip([0, 1], ["No Disease", "Disease"], PALETTE):
    sub = df[df["target"] == t]
    ax.scatter(sub["age"], sub["thalach"], alpha=0.5, s=20, label=lbl, color=c)
ax.set_title("Maximum Heart Rate vs Age", fontsize=14, fontweight="bold")
ax.set_xlabel("Age"); ax.set_ylabel("Max Heart Rate (bpm)"); ax.legend()
save("06_max_heart_rate_vs_age")

# ── 7. Sex Distribution ───────────────────────────────────────────────────────
sex_map = {0: "Female", 1: "Male"}
df["sex_label"] = df["sex"].map(sex_map)
ct2 = pd.crosstab(df["sex_label"], df["target"])
ct2.columns = ["No Disease", "Disease"]
fig, ax = plt.subplots(figsize=(6, 4))
ct2.plot(kind="bar", ax=ax, color=PALETTE, edgecolor="white")
ax.set_title("Sex vs Heart Disease", fontsize=14, fontweight="bold")
ax.set_xlabel("Sex"); ax.set_ylabel("Count")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(title="Status")
save("07_sex_vs_target")

# ── 8. Blood Pressure Distribution ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(data=df, x="age_group", y="trestbps", hue="target",
            palette=PALETTE, order=["<40", "40-50", "50-60", "60+"], ax=ax)
ax.set_title("Blood Pressure by Age Group & Disease Status", fontsize=14, fontweight="bold")
ax.set_xlabel("Age Group"); ax.set_ylabel("Resting Blood Pressure (mmHg)")
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles, ["No Disease", "Disease"], title="Status")
save("08_bp_by_age_group")

print("\n[✓] All 8 visualizations generated in static/charts/")
