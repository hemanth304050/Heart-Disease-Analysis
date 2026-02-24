"""
Flask Web Application - Heart Disease Analysis Dashboard
Full-stack web integration with Bootstrap UI, interactive charts, and ML prediction.
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json, pickle, os, sqlite3

app = Flask(__name__)

def load_artifacts():
    """Load model, scaler, features, and metrics from saved files."""
    with open("data/model.pkl",   "rb") as f: model   = pickle.load(f)
    with open("data/scaler.pkl",  "rb") as f: scaler  = pickle.load(f)
    with open("data/features.pkl","rb") as f: features= pickle.load(f)
    with open("data/metrics.json","r")  as f: metrics = json.load(f)
    return model, scaler, features, metrics

def get_stats():
    conn = sqlite3.connect("data/heart_disease.db")
    df   = pd.read_sql("SELECT * FROM heart_data", conn); conn.close()
    return {
        "total":     len(df),
        "disease":   int(df["target"].sum()),
        "no_disease":int((df["target"] == 0).sum()),
        "avg_age":   round(df["age"].mean(), 1),
        "avg_chol":  round(df["chol"].mean(), 1),
        "avg_bp":    round(df["trestbps"].mean(), 1),
        "avg_hr":    round(df["thalach"].mean(), 1),
        "disease_pct": round(df["target"].mean() * 100, 1),
    }

# ── Pages ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    stats = get_stats()
    return render_template("index.html", stats=stats)

@app.route("/dashboard")
def dashboard():
    stats = get_stats()
    return render_template("dashboard.html", stats=stats)

@app.route("/visualizations")
def visualizations():
    charts = sorted([
        f for f in os.listdir("static/charts")
        if f.endswith(".png") and f[:2].isdigit() and int(f[:2]) <= 8
    ])
    titles = [
        "Target Distribution", "Age Distribution by Status",
        "Correlation Heatmap", "Chest Pain vs Disease",
        "Cholesterol vs Age", "Max Heart Rate vs Age",
        "Sex vs Disease", "Blood Pressure by Age Group"
    ]
    chart_data = list(zip(charts, titles))
    return render_template("visualizations.html", chart_data=chart_data)

@app.route("/performance")
def performance():
    _, _, _, metrics = load_artifacts()
    perf_charts = sorted([
        f for f in os.listdir("static/charts")
        if f.endswith(".png") and f[:2].isdigit() and int(f[:2]) >= 9
    ])
    return render_template("performance.html", metrics=metrics, charts=perf_charts)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    probability = None
    if request.method == "POST":
        model, scaler, features, _ = load_artifacts()
        data = {
            "age":      int(request.form["age"]),
            "sex":      int(request.form["sex"]),
            "cp":       int(request.form["cp"]),
            "trestbps": int(request.form["trestbps"]),
            "chol":     int(request.form["chol"]),
            "fbs":      int(request.form["fbs"]),
            "restecg":  int(request.form["restecg"]),
            "thalach":  int(request.form["thalach"]),
            "exang":    int(request.form["exang"]),
            "oldpeak":  float(request.form["oldpeak"]),
            "slope":    int(request.form["slope"]),
            "ca":       int(request.form["ca"]),
            "thal":     int(request.form["thal"]),
        }
        X = pd.DataFrame([data])[features]
        Xs = scaler.transform(X)
        prediction  = int(model.predict(Xs)[0])
        probability = round(float(model.predict_proba(Xs)[0][1]) * 100, 1)
    return render_template("predict.html", prediction=prediction, probability=probability)

@app.route("/story")
def story():
    stats = get_stats()
    return render_template("story.html", stats=stats)

@app.route("/api/stats")
def api_stats():
    return jsonify(get_stats())

@app.route("/api/chart-data")
def api_chart_data():
    conn = sqlite3.connect("data/heart_disease.db")
    df   = pd.read_sql("SELECT * FROM heart_data", conn); conn.close()
    cp_map = {0:"Typical Angina",1:"Atypical Angina",2:"Non-Anginal",3:"Asymptomatic"}
    age_bins  = pd.cut(df["age"], bins=[0,40,50,60,100], labels=["<40","40-50","50-60","60+"])
    age_group = df.groupby([age_bins, "target"])["age"].count().unstack(fill_value=0)
    return jsonify({
        "target":  df["target"].value_counts().to_dict(),
        "cp":      df.groupby("cp")["target"].mean().round(3).to_dict(),
        "age_chol":{"age": df["age"].tolist(), "chol": df["chol"].tolist(), "target": df["target"].tolist()},
        "age_groups": {
            "labels": [str(l) for l in age_group.index.tolist()],
            "no_disease": age_group.get(0, pd.Series(0, index=age_group.index)).tolist(),
            "disease":    age_group.get(1, pd.Series(0, index=age_group.index)).tolist(),
        }
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
