"""
Data Preparation - Heart Disease Analysis
Cleans, preprocesses, and feature-engineers the raw dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os, pickle

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv("data/heart_disease.csv")
print(f"Raw shape : {df.shape}")

# ── 1. Missing values ─────────────────────────────────────────────────────────
print("\nMissing values:\n", df.isnull().sum())
df.dropna(inplace=True)

# ── 2. Duplicate rows ─────────────────────────────────────────────────────────
before = len(df)
df.drop_duplicates(inplace=True)
print(f"\nDuplicates removed : {before - len(df)}")

# ── 3. Outlier capping (IQR) ─────────────────────────────────────────────────
numeric_cols = ["trestbps", "chol", "thalach", "oldpeak"]
for col in numeric_cols:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

# ── 4. Feature engineering ───────────────────────────────────────────────────
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 40, 50, 60, 100],
    labels=["<40", "40-50", "50-60", "60+"]
)
df["chol_risk"]   = (df["chol"] > 240).astype(int)
df["bp_risk"]     = (df["trestbps"] > 130).astype(int)
df["hr_efficiency"] = df["thalach"] / df["age"]

# ── 5. Encode categoricals ───────────────────────────────────────────────────
le = LabelEncoder()
df["age_group_enc"] = le.fit_transform(df["age_group"])

# ── 6. Scale numeric features ────────────────────────────────────────────────
scaler = StandardScaler()
scale_cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "hr_efficiency"]
df_scaled = df.copy()
df_scaled[scale_cols] = scaler.fit_transform(df[scale_cols])

os.makedirs("data", exist_ok=True)
df.to_csv("data/heart_clean.csv", index=False)
df_scaled.to_csv("data/heart_scaled.csv", index=False)

with open("data/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print(f"\nCleaned shape  : {df.shape}")
print("\nProcessed data sample:")
print(df[["age", "chol", "trestbps", "thalach", "target", "age_group", "chol_risk", "bp_risk"]].head())
print(f"\n[✓] Saved → data/heart_clean.csv")
print(f"[✓] Saved → data/heart_scaled.csv")
print(f"[✓] Saved → data/scaler.pkl")

# ── 7. Summary statistics ─────────────────────────────────────────────────────
print("\nDescriptive Statistics:")
print(df.describe().round(2))
