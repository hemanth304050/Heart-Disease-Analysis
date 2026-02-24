# 🫀 Heart Disease Analysis — SmartBridge Project

> **Category:** Data Analytics with Tableau  
> **Level:** Intermediate  
> **Skills:** Python · Data Preprocessing · Bootstrap · Flask · Scikit-learn

---

## 📁 Project Structure

```
smartbridge/
├── generate_dataset.py     # Step 1 – Data Collection & SQLite DB
├── data_preparation.py     # Step 2 – Cleaning, Feature Engineering
├── data_visualization.py   # Step 3 – 8 Matplotlib/Seaborn Charts
├── performance_testing.py  # Step 4 – Train 4 ML Models + Metrics
├── app.py                  # Flask Web Application
├── setup.py                # ⭐ Run this FIRST (one-time setup)
├── requirements.txt        # Python dependencies
├── templates/              # HTML pages (Bootstrap dark UI)
│   ├── base.html
│   ├── index.html
│   ├── dashboard.html
│   ├── visualizations.html
│   ├── performance.html
│   ├── predict.html
│   └── story.html
├── static/charts/          # Generated chart PNGs (auto-created)
└── data/                   # Dataset, DB, model, metrics (auto-created)
    ├── heart_disease.csv
    ├── heart_disease.db
    ├── heart_clean.csv
    ├── heart_scaled.csv
    ├── model.pkl
    ├── scaler.pkl
    ├── features.pkl
    └── metrics.json
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run one-time setup (generates data, trains models, creates charts)
```bash
python setup.py
```

### 3. Launch the web app
```bash
python app.py
```

### 4. Open browser
```
http://127.0.0.1:5000
```

---

## 📊 Web App Pages

| Page | URL | Description |
|------|-----|-------------|
| Home | `/` | Project overview & KPI cards |
| Dashboard | `/dashboard` | Live Chart.js analytics dashboard |
| Visualizations | `/visualizations` | Gallery of 8 data charts |
| Performance | `/performance` | ML model comparison & ROC curves |
| Predict | `/predict` | Real-time heart disease prediction |
| Story | `/story` | Data narrative & recommendations |

---

## 🤖 Machine Learning Models

| Model | Features |
|-------|----------|
| Logistic Regression | Baseline classifier |
| Random Forest | Ensemble, feature importance |
| Gradient Boosting | Best accuracy (typically) |
| SVM | Support vector classification |

**Evaluation:** Accuracy, AUC, F1, Precision, Recall, 5-Fold CV

---

## 📋 Dataset Features (13 Clinical Variables)

| Feature | Description |
|---------|-------------|
| age | Patient age in years |
| sex | 0=Female, 1=Male |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mmHg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting ECG results |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels colored by fluoroscopy |
| thal | Thalassemia type |
| **target** | **0 = No Disease, 1 = Disease** |

---

## 🎓 SmartBridge Module Mapping

| Module | File(s) |
|--------|---------|
| Data Collection & Extraction | `generate_dataset.py` |
| Data Preparation | `data_preparation.py` |
| Data Visualization | `data_visualization.py` |
| Dashboard | `app.py` + `templates/dashboard.html` |
| Story | `templates/story.html` |
| Performance Testing | `performance_testing.py` |
| Web Integration | `app.py` + all templates |
| Project Demonstration | This README + the web app |
