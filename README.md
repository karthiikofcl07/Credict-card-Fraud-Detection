# 🛡️ Credit Card Fraud Detection — System Description

> **Task:** Build a supervised machine learning model to classify credit card transactions as **fraudulent** or **legitimate**, using a dataset of 1.85M real-world transactions. Experiment with Logistic Regression, Decision Trees, and Random Forests.

---

## 📌 Problem Statement

Credit card fraud costs the global economy **$32 billion annually**. Traditional rule-based systems generate excessive false positives, frustrating legitimate customers. Machine learning enables pattern detection at scale — catching fraud in milliseconds without blocking real purchases.

**Key challenge:** Extreme class imbalance (~0.5% fraud rate) requiring careful handling via class weighting.

---

## 🗂️ Dataset Overview

| Property | Value |
|---|---|
| **Source** | Synthetic credit card transactions (Jan 2019–Dec 2020) |
| **Train Rows** | 1,296,675 |
| **Test Rows** | 555,719 |
| **Total** | 1,851,394 transactions |
| **Fraud (Train)** | 7,506 (0.58%) |
| **Features** | 23 raw → 16 engineered |
| **Target** | `is_fraud` (0 = legitimate, 1 = fraud) |

---

## 🔄 ML Pipeline Flowchart

```
┌──────────────────────────────────────────────────────────────────┐
│                     RAW DATA (fraudTrain.csv)                    │
│         1,296,675 transactions  ·  23 columns                    │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                  DATA LOADING & VALIDATION                        │
│   • Read CSV with pandas                                          │
│   • Check dtypes & null values                                    │
│   • Verify target column distribution                             │
│   • Stratified sampling (all fraud + 5% legitimate)              │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│              EXPLORATORY DATA ANALYSIS (EDA)                      │
│                                                                   │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│   │ Class Dist  │  │ Amount Dist │  │ Temporal Patterns       │ │
│   │ (imbalance) │  │ by fraud    │  │ (hour / day patterns)   │ │
│   └─────────────┘  └─────────────┘  └─────────────────────────┘ │
│                                                                   │
│   ┌─────────────────────────┐  ┌──────────────────────────────┐  │
│   │ Category Fraud Rates    │  │ Correlation Heatmap          │  │
│   └─────────────────────────┘  └──────────────────────────────┘  │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING                             │
│                                                                   │
│   Raw Features              Engineered Features                   │
│   ────────────              ────────────────────                  │
│   trans_datetime     ──►    hour, day_of_week, month             │
│                      ──►    is_weekend, is_night                 │
│   dob                ──►    age (years)                          │
│   lat/long           ──►    distance (haversine approx)          │
│   merch_lat/long     ──►    (used in distance calc)              │
│   gender             ──►    gender_enc (binary)                  │
│   category           ──►    category_enc (label encoded)         │
│   amt                ──►    amt_log (log1p transform)            │
│   city_pop                  city_pop (kept as-is)                │
│                                                                   │
│   Final feature vector: 16 dimensions                            │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                    DATA SPLITTING & SCALING                       │
│                                                                   │
│   Train set (71,964)        Test set (29,823)                    │
│   ─────────────────         ─────────────────                    │
│   X_train, y_train          X_test, y_test                       │
│                                                                   │
│   StandardScaler fit on train → transform both sets              │
│   (used only for Logistic Regression)                            │
└──────────────────────┬───────────────────────────────────────────┘
                       │
           ┌───────────┼────────────┐
           ▼           ▼            ▼
    ┌────────────┐ ┌─────────┐ ┌──────────────┐
    │  Logistic  │ │Decision │ │   Random     │
    │ Regression │ │  Tree   │ │   Forest     │
    │            │ │         │ │              │
    │ C=1.0      │ │depth=10 │ │ n=100        │
    │ balanced   │ │balanced │ │ depth=12     │
    │ max_iter   │ │         │ │ n_jobs=-1    │
    │  =500      │ │         │ │ balanced     │
    └─────┬──────┘ └────┬────┘ └──────┬───────┘
          │             │             │
          └──────┬───────────────────┘
                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                       MODEL EVALUATION                            │
│                                                                   │
│   Metric           LR          DT          RF (Best)             │
│   ─────────────────────────────────────────────────              │
│   AUC-ROC          0.9364      0.9828      0.9941 ★              │
│   Avg Precision    0.6992      0.9363      0.9604                 │
│   Fraud F1         0.5924      0.8295      0.9048                 │
│   Accuracy         92.33%      97.15%      98.59%                │
│                                                                   │
│   Plots generated:                                               │
│   • ROC Curves           • Precision-Recall Curves               │
│   • Confusion Matrices   • Feature Importance                    │
│   • Model Comparison     • Age/Amount/Category distributions     │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                    MODEL SERIALIZATION                            │
│                                                                   │
│   ✅  Best model (Random Forest) → outputs/best_model.pkl        │
│   ✅  StandardScaler             → included in pickle bundle      │
│   ✅  Feature list               → included in pickle bundle      │
│   ✅  Results summary            → outputs/results_summary.json  │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                   FLASK WEB APPLICATION                           │
│                                                                   │
│   POST /predict        → Single transaction inference            │
│   POST /api/batch      → Batch prediction (up to 100 txns)       │
│   GET  /api/stats      → Dataset & model statistics              │
│   GET  /api/model_metrics → Per-model performance table          │
│   GET  /health         → Service health check                    │
│   GET  /               → Responsive dashboard UI                 │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Full System Architecture

```
fraud_detection/
├── app.py                         ← Flask REST API + ML inference
├── FraudDetection_Analysis.ipynb  ← Full EDA + training notebook
├── description.md                 ← This document
├── README.md                      ← GitHub documentation
├── requirements.txt               ← Python dependencies
├── templates/
│   └── index.html                 ← Responsive dashboard UI
├── outputs/
│   ├── best_model.pkl             ← Serialized Random Forest
│   ├── results_summary.json       ← Model metrics
│   ├── class_distribution.png
│   ├── amount_category.png
│   ├── temporal_patterns.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   ├── confusion_matrices.png
│   ├── roc_pr_curves.png
│   └── model_comparison.png
└── static/                        ← (for additional assets)
```

---

## 🧠 Algorithms Compared

### 1. Logistic Regression
- Baseline linear classifier with L2 regularisation
- Features scaled with StandardScaler
- `class_weight='balanced'` to handle imbalance
- AUC-ROC: **0.9364**

### 2. Decision Tree
- Non-linear, interpretable tree with depth=10
- Captures complex feature interactions
- Fast training and inference
- AUC-ROC: **0.9828**

### 3. Random Forest ⭐ Best
- Ensemble of 100 decision trees
- Robust to overfitting via bagging
- Returns calibrated probabilities
- AUC-ROC: **0.9941** | F1: **0.9048**

---

## 📈 Key Findings

| Finding | Insight |
|---|---|
| **Amount is the top predictor** | Fraudulent transactions average $527 vs $67 for legitimate |
| **Late night is risky** | Fraud rate peaks between 10 PM–3 AM |
| **misc_net category leads** | 72% fraud rate — likely CNP (Card Not Present) fraud |
| **Geographic distance matters** | Large card-holder to merchant distance signals anomaly |
| **Class imbalance is severe** | Only 0.52% fraud — requires balanced class weighting |

---

## 🚀 Running the App

```bash
# Install dependencies
pip install -r requirements.txt

# Start Flask server
python app.py

# Open in browser
http://localhost:5000
```

---

*Built with Python 3.11 · scikit-learn · pandas · matplotlib · Flask*
