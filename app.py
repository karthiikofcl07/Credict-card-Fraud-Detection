"""
╔══════════════════════════════════════════════════════════════╗
║       FraudShield AI — Credit Card Fraud Detection          ║
║       Flask Backend  |  REST API  |  ML Inference           ║
╚══════════════════════════════════════════════════════════════╝
"""

import os, json, pickle, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# ─────────────────────────────────────────────
#  Model Loading
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "outputs", "best_model.pkl")
_model_bundle = None

def load_model():
    global _model_bundle
    try:
        with open(MODEL_PATH, "rb") as f:
            _model_bundle = pickle.load(f)
        print(f"Model loaded: {MODEL_PATH}")
    except FileNotFoundError:
        print("Model not found — running in heuristic-demo mode")
        _model_bundle = None

load_model()

# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
CATEGORIES = [
    "entertainment","food_dining","gas_transport","grocery_net",
    "grocery_pos","health_fitness","home","kids_pets",
    "misc_net","misc_pos","personal_care","shopping_net",
    "shopping_pos","travel",
]

CAT_RISK = {
    "misc_net":0.72,"grocery_pos":0.65,"misc_pos":0.58,
    "shopping_net":0.54,"travel":0.48,"entertainment":0.38,
    "food_dining":0.32,"health_fitness":0.28,"kids_pets":0.25,
    "personal_care":0.20,"gas_transport":0.18,"grocery_net":0.15,
    "shopping_pos":0.12,"home":0.08,
}

FEATURES = [
    "amt","amt_log","hour","day_of_week","month",
    "is_weekend","is_night","age","distance","city_pop",
    "gender_enc","category_enc","lat","long","merch_lat","merch_long",
]

_le = LabelEncoder()
_le.fit(CATEGORIES)

# ─────────────────────────────────────────────
#  Feature Engineering
# ─────────────────────────────────────────────
def build_features(d):
    try:
        ts = pd.to_datetime(d.get("trans_datetime", datetime.now().isoformat()))
    except Exception:
        ts = datetime.now()

    try:
        dob = pd.to_datetime(d.get("dob", "1980-01-01"))
        age = max(0, (ts - dob).days // 365)
    except Exception:
        age = 40

    amt      = float(d.get("amt", 0) or 0)
    lat      = float(d.get("lat", 36.0)  or 36.0)
    lon      = float(d.get("long", -80.0) or -80.0)
    mlat     = float(d.get("merch_lat", 36.0)  or 36.0)
    mlon     = float(d.get("merch_long", -80.0) or -80.0)
    city_pop = float(d.get("city_pop", 5000) or 5000)
    gender   = d.get("gender", "F")
    cat      = d.get("category", "misc_net")
    distance = float(np.sqrt((lat - mlat)**2 + (lon - mlon)**2))

    try:
        cat_enc = int(_le.transform([cat])[0])
    except Exception:
        cat_enc = 0

    return np.array([
        amt, np.log1p(amt),
        ts.hour, ts.dayofweek, ts.month,
        int(ts.dayofweek >= 5),
        int(ts.hour >= 22 or ts.hour <= 6),
        age, distance, city_pop,
        int(gender == "M"),
        cat_enc, lat, lon, mlat, mlon,
    ]).reshape(1, -1)


# ─────────────────────────────────────────────
#  Inference
# ─────────────────────────────────────────────
def predict_fraud(d):
    feats = build_features(d)
    amt   = float(d.get("amt", 0) or 0)
    cat   = d.get("category", "misc_net")

    if _model_bundle is not None:
        prob       = float(_model_bundle["model"].predict_proba(feats)[0][1])
        model_name = "Random Forest"
    else:
        base = CAT_RISK.get(cat, 0.3)
        if amt > 800:
            base = min(base + 0.30, 0.97)
        elif amt > 400:
            base = min(base + 0.15, 0.90)
        prob       = float(np.clip(base + np.random.uniform(-0.04, 0.04), 0.01, 0.99))
        model_name = "Heuristic Demo"

    risk = (
        "CRITICAL" if prob >= 0.85 else
        "HIGH"     if prob >= 0.65 else
        "MEDIUM"   if prob >= 0.35 else
        "LOW"
    )

    try:
        ts   = pd.to_datetime(d.get("trans_datetime", datetime.now().isoformat()))
        lat  = float(d.get("lat", 36.0)  or 36.0)
        lon  = float(d.get("long", -80.0) or -80.0)
        mlat = float(d.get("merch_lat", 36.0)  or 36.0)
        mlon = float(d.get("merch_long", -80.0) or -80.0)
        dist = float(np.sqrt((lat - mlat)**2 + (lon - mlon)**2))
    except Exception:
        ts = datetime.now(); dist = 0.0

    factors = []
    if amt > 800:
        factors.append({"factor":"High Transaction Amount","impact":"high",
                        "detail":f"${amt:,.2f} is far above the dataset average of ~$67"})
    elif amt > 400:
        factors.append({"factor":"Above-Average Amount","impact":"medium",
                        "detail":f"${amt:,.2f} is notably above the mean transaction"})

    if ts.hour >= 22 or ts.hour <= 5:
        factors.append({"factor":"Late-Night Transaction","impact":"high",
                        "detail":f"Time {ts.strftime('%H:%M')} falls in the 10 PM–5 AM fraud peak"})

    cat_r = CAT_RISK.get(cat, 0.3)
    if cat_r >= 0.55:
        factors.append({"factor":"High-Risk Merchant Category","impact":"high",
                        "detail":f"'{cat.replace('_',' ').title()}' has {cat_r*100:.0f}% elevated fraud rate"})
    elif cat_r >= 0.35:
        factors.append({"factor":"Moderate-Risk Category","impact":"medium",
                        "detail":f"'{cat.replace('_',' ').title()}' carries moderate fraud risk"})

    if dist > 8:
        factors.append({"factor":"Large Geographic Distance","impact":"high",
                        "detail":f"Merchant is ~{dist*111:.0f} km from cardholder location"})
    elif dist > 3:
        factors.append({"factor":"Geographic Mismatch","impact":"medium",
                        "detail":f"Card used {dist:.1f} degrees from home location"})

    try:
        cpop = int(d.get("city_pop", 9999) or 9999)
        if cpop < 1000:
            factors.append({"factor":"Rural Merchant Location","impact":"low",
                            "detail":"Very small city — unusual merchant location pattern"})
    except Exception:
        pass

    if not factors:
        factors.append({"factor":"No Significant Risk Signals","impact":"low",
                        "detail":"Transaction patterns are consistent with legitimate behaviour"})

    return {
        "fraud_probability" : round(prob * 100, 2),
        "fraud_score"       : round(prob, 6),
        "prediction"        : "FRAUD" if prob >= 0.5 else "LEGITIMATE",
        "risk_level"        : risk,
        "confidence"        : round(abs(prob - 0.5) * 200, 1),
        "risk_factors"      : factors,
        "model_used"        : model_name,
        "timestamp"         : datetime.now().isoformat(),
    }


# ─────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",
                           categories=CATEGORIES,
                           model_loaded=_model_bundle is not None)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        result = predict_fraud(request.get_json(force=True))
        return jsonify({"status": "success", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/stats")
def api_stats():
    stats = {
        "total_transactions":"1,851,394","fraud_transactions":"9,651",
        "fraud_rate":"0.52%","best_model":"Random Forest",
        "best_auc":"0.9941","best_f1":"0.9048",
        "models_trained":3,"features_used":16,
        "dataset_period":"Jan 2019 – Dec 2020","accuracy":"98.59%",
    }
    if _model_bundle and "results_summary" in _model_bundle:
        rf = _model_bundle["results_summary"].get("Random Forest", {})
        stats["best_auc"] = str(rf.get("roc_auc","0.9941"))
        stats["best_f1"]  = str(rf.get("fraud_f1","0.9048"))
    return jsonify(stats)


@app.route("/api/model_metrics")
def api_model_metrics():
    if _model_bundle and "results_summary" in _model_bundle:
        return jsonify(_model_bundle["results_summary"])
    return jsonify({
        "Logistic Regression":{"roc_auc":0.9364,"avg_precision":0.6992,
                                "fraud_precision":0.4794,"fraud_recall":0.7753,
                                "fraud_f1":0.5924,"accuracy":0.9233},
        "Decision Tree":      {"roc_auc":0.9828,"avg_precision":0.9363,
                                "fraud_precision":0.7281,"fraud_recall":0.9636,
                                "fraud_f1":0.8295,"accuracy":0.9715},
        "Random Forest":      {"roc_auc":0.9941,"avg_precision":0.9604,
                                "fraud_precision":0.8805,"fraud_recall":0.9305,
                                "fraud_f1":0.9048,"accuracy":0.9859},
    })


@app.route("/api/batch_predict", methods=["POST"])
def batch_predict():
    try:
        body  = request.get_json(force=True)
        txns  = body.get("transactions", [])[:100]
        if not txns:
            return jsonify({"status":"error","message":"No transactions provided"}), 400
        results   = []
        for i, t in enumerate(txns):
            r = predict_fraud(t)
            r["transaction_id"] = t.get("trans_num", f"TXN{i+1:04d}")
            results.append(r)
        fraud_cnt = sum(1 for r in results if r["prediction"] == "FRAUD")
        return jsonify({"status":"success","total":len(results),
                        "fraud_detected":fraud_cnt,"legitimate":len(results)-fraud_cnt,
                        "fraud_rate_pct":round(fraud_cnt/len(results)*100,2),
                        "results":results})
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status":"healthy","model_loaded":_model_bundle is not None,
                    "version":"1.0.0","timestamp":datetime.now().isoformat()})


if __name__ == "__main__":
    print("\nFraudShield AI  →  http://localhost:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
