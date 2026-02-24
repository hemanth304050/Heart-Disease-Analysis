"""
Data Collection & Extraction - Heart Disease Analysis
Generates a realistic dataset and stores it in SQLite database.
"""

import pandas as pd
import numpy as np
import sqlite3
import os

np.random.seed(42)
n = 1000

# ── Generate synthetic but realistic heart disease dataset ──────────────────
age     = np.random.randint(29, 77, n)
sex     = np.random.choice([0, 1], n, p=[0.32, 0.68])          # 0=Female 1=Male
cp      = np.random.choice([0, 1, 2, 3], n, p=[0.47, 0.17, 0.28, 0.08])  # chest-pain type
trestbps= np.clip(np.random.normal(131, 17, n).astype(int), 94, 200)
chol    = np.clip(np.random.normal(246, 51, n).astype(int), 126, 564)
fbs     = np.random.choice([0, 1], n, p=[0.85, 0.15])          # fasting blood sugar >120
restecg = np.random.choice([0, 1, 2], n, p=[0.48, 0.49, 0.03])
thalach = np.clip(np.random.normal(149, 22, n).astype(int), 71, 202)
exang   = np.random.choice([0, 1], n, p=[0.68, 0.32])
oldpeak = np.clip(np.round(np.random.exponential(1.0, n), 1), 0.0, 6.2)
slope   = np.random.choice([0, 1, 2], n, p=[0.07, 0.47, 0.46])
ca      = np.random.choice([0, 1, 2, 3], n, p=[0.59, 0.22, 0.13, 0.06])
thal    = np.random.choice([1, 2, 3], n, p=[0.06, 0.55, 0.39])

# target influenced by risk factors
risk = (
    0.02 * (age - 50)
    + 0.4  * sex
    + 0.5  * (cp == 0).astype(int)
    - 0.3  * (cp == 3).astype(int)
    + 0.3  * exang
    + 0.4  * oldpeak
    - 0.01 * (thalach - 150)
)
prob   = 1 / (1 + np.exp(-risk))
target = (np.random.rand(n) < prob).astype(int)

df = pd.DataFrame({
    "age": age, "sex": sex, "cp": cp,
    "trestbps": trestbps, "chol": chol, "fbs": fbs,
    "restecg": restecg, "thalach": thalach, "exang": exang,
    "oldpeak": oldpeak, "slope": slope, "ca": ca,
    "thal": thal, "target": target
})

os.makedirs("data", exist_ok=True)
csv_path = os.path.join("data", "heart_disease.csv")
df.to_csv(csv_path, index=False)
print(f"[✓] Dataset saved → {csv_path}  ({len(df)} rows)")

# ── Store in SQLite ─────────────────────────────────────────────────────────
db_path = os.path.join("data", "heart_disease.db")
conn    = sqlite3.connect(db_path)
df.to_sql("heart_data", conn, if_exists="replace", index=False)
conn.close()
print(f"[✓] SQLite database saved → {db_path}")

# ── Quick verification: extract from DB ─────────────────────────────────────
conn   = sqlite3.connect(db_path)
sample = pd.read_sql("SELECT * FROM heart_data LIMIT 5", conn)
conn.close()
print("\nSample rows extracted from DB:")
print(sample.to_string(index=False))
print(f"\nShape : {df.shape}")
print(f"Target distribution:\n{df['target'].value_counts().rename({0:'No Disease',1:'Disease'})}")
