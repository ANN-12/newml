"""
app.py — Prediction server | 33 features
Run: python app.py

FIXES APPLIED:
  1. Feature list updated to 33 (matches train.py exactly)
  2. Input validation added (negative values, suspiciously large values)
  3. Metadata loaded and shown on startup
  4. Health endpoint now shows last trained date + accuracy
  5. Better error messages
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import os
import json
from collections import Counter
from datetime import datetime

app = Flask(__name__)
CORS(app)

MODEL_DIR = "models"

# ── MUST match train.py exactly — 33 features ──
ALL_FEATURES = [
    "dwell_mean", "dwell_std", "dwell_cv",
    "flight_mean", "flight_std", "flight_cv",
    "timing_entropy", "total_duration", "wpm",
    "dg_th", "dg_he", "dg_qu", "dg_br", "dg_ow",
    "dg_fo", "dg_ju", "dg_ov", "dg_er", "dg_la", "dg_sp",
    "tg_the", "tg_bro", "tg_own", "tg_ove", "tg_ver",
    "kd_e", "kd_o", "kd_t", "kd_h", "kd_r", "kd_u", "kd_space",
    "backspace_rate",
]

# ── FEATURE VALIDATION LIMITS ──
# Values outside these ranges are rejected as invalid input
FEATURE_LIMITS = {
    "dwell_mean":      (10,    500),
    "dwell_std":       (0,     300),
    "dwell_cv":        (0,     5),
    "flight_mean":     (0,     2000),
    "flight_std":      (0,     2000),
    "flight_cv":       (0,     10),
    "timing_entropy":  (0,     10),
    "total_duration":  (3000,  60000),
    "wpm":             (1,     200),
    "backspace_rate":  (0,     1),
}

# ── LOAD MODELS ──
print("\n🔄 Loading models ...")
try:
    trained_models  = pickle.load(open(f"{MODEL_DIR}/trained_models.pkl",  "rb"))
    model_scalers   = pickle.load(open(f"{MODEL_DIR}/model_scalers.pkl",   "rb"))
    le              = pickle.load(open(f"{MODEL_DIR}/label_encoder.pkl",   "rb"))
    model_feat_idxs = pickle.load(open(f"{MODEL_DIR}/model_feat_idxs.pkl","rb"))
except FileNotFoundError as e:
    print(f"❌ Model file missing: {e}")
    print("   Run: python train.py  first!")
    exit(1)

# ── FIX 3: LOAD METADATA ──
metadata = {}
metadata_path = f"{MODEL_DIR}/metadata.json"
if os.path.exists(metadata_path):
    metadata = json.load(open(metadata_path))
    print(f"[OK] {len(trained_models)} models loaded")
    print(f"[OK] Users        : {list(le.classes_)}")
    print(f"[OK] Features     : {len(ALL_FEATURES)}")
    print(f"[OK] Trained on   : {metadata.get('trained_on', 'unknown')[:19]}")
    print(f"[OK] Total samples: {metadata.get('total_samples', 'unknown')}")
    print(f"[OK] Best model   : {metadata.get('best_model','unknown')} "
          f"({metadata.get('best_accuracy', 0)*100:.1f}%)")
    print(f"[OK] Avg accuracy : {metadata.get('avg_accuracy', 0)*100:.1f}%")
else:
    print(f"[OK] {len(trained_models)} models | users: {list(le.classes_)}")
    print(f"⚠️  metadata.json not found — run updated train.py")


# ═══════════════════════════════════════════════════════════
# FIX 2 — INPUT VALIDATION FUNCTION
# ═══════════════════════════════════════════════════════════
def validate_features(features):
    """
    Checks each feature value is within expected range.
    Returns (is_valid, error_message)
    """
    for feat, (min_val, max_val) in FEATURE_LIMITS.items():
        if feat not in features:
            continue
        val = features[feat]

        if val < 0 and feat != "backspace_rate":
            return False, f"Negative value not allowed for '{feat}': {val}"

        if val < min_val:
            return False, (
                f"Value too small for '{feat}': {val:.2f} "
                f"(min expected: {min_val})"
            )

        if val > max_val:
            return False, (
                f"Value too large for '{feat}': {val:.2f} "
                f"(max expected: {max_val})"
            )

    return True, None


# ═══════════════════════════════════════════════════════════
# PREDICTION LOGIC — WEIGHTED VOTING
# ═══════════════════════════════════════════════════════════
def predict_with_voting(features_dict, top_k=3):
    full_vec           = np.array([[features_dict[f] for f in ALL_FEATURES]])
    appearance_counter = Counter()
    per_model_top3     = {}

    for name, model in trained_models.items():
        col_idx   = model_feat_idxs[name]
        x_sc      = model_scalers[name].transform(full_vec[:, col_idx])
        proba     = model.predict_proba(x_sc)[0]
        top_idx   = np.argsort(proba)[::-1][:top_k]
        top_users = le.inverse_transform(top_idx).tolist()
        top_probs = proba[top_idx].tolist()

        per_model_top3[name] = [
            {"user": u, "confidence": round(p * 100, 2), "rank": i + 1}
            for i, (u, p) in enumerate(zip(top_users, top_probs))
        ]
        for user in top_users:
            appearance_counter[user] += 1

    # Weighted score — rank 1 gets more weight than rank 2 or 3
    weighted = Counter()
    for name, top3 in per_model_top3.items():
        for item in top3:
            weighted[item["user"]] += (top_k + 1 - item["rank"])

    winner = sorted(
        appearance_counter.keys(),
        key=lambda u: (appearance_counter[u], weighted[u]),
        reverse=True
    )[0]

    # Calculate confidence percentage
    confidence_pct = round(
        appearance_counter[winner] / (top_k * len(trained_models)) * 100, 1
    )

    return {
        "winner":             winner,
        "confidence_pct":     confidence_pct,
        "winner_appearances": appearance_counter[winner],
        "total_models":       len(trained_models),
        "max_appearances":    top_k * len(trained_models),
        "appearance_counts":  dict(appearance_counter.most_common()),
        "weighted_scores":    dict(weighted.most_common()),
        "per_model_top3":     per_model_top3,
        "timestamp":          datetime.utcnow().isoformat(),
    }


# ═══════════════════════════════════════════════════════════
# ROUTE 1 — /predict   POST
# ═══════════════════════════════════════════════════════════
@app.route("/predict", methods=["POST"])
def predict():
    # Check JSON body exists
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body sent"}), 400

    # Check all features present
    missing = [f for f in ALL_FEATURES if f not in data]
    if missing:
        return jsonify({
            "error":   "Missing features in request",
            "missing": missing
        }), 400

    try:
        # Convert all values to float
        features = {f: float(data[f]) for f in ALL_FEATURES}

        # FIX 2 — Validate feature values before predicting
        is_valid, error_msg = validate_features(features)
        if not is_valid:
            return jsonify({
                "error": f"Invalid feature value: {error_msg}"
            }), 400

        # Run prediction
        result = predict_with_voting(features)

        # ── TERMINAL LOGGING ──
        print(f"\n🔐 AUTHENTICATION RESULT:")
        print(f"   ├─ Predicted  : {result['winner'].upper()}")
        print(f"   ├─ Confidence : {result['confidence_pct']}%  "
              f"({result['winner_appearances']}/{result['max_appearances']} votes)")
        print(f"   ├─ Dwell Time : {features['dwell_mean']:.0f}ms  "
              f"|  Flight Time: {features['flight_mean']:.0f}ms")
        print(f"   ├─ WPM        : {features['wpm']:.0f}  "
              f"|  Duration: {features['total_duration']/1000:.1f}s")
        print(f"   └─ Timestamp  : {result['timestamp'][:19]} UTC")
        print("=" * 60)

        return jsonify(result), 200

    except ValueError as e:
        return jsonify({"error": f"Non-numeric value in features: {str(e)}"}), 400
    except Exception as e:
        print(f"✗ PREDICTION ERROR: {str(e)}")
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════
# ROUTE 2 — /health   GET
# ═══════════════════════════════════════════════════════════
@app.route("/health", methods=["GET"])
def health():
    # FIX 4 — Health now returns full system info
    return jsonify({
        "status":         "ok",
        "models_loaded":  len(trained_models),
        "users":          list(le.classes_),
        "num_features":   len(ALL_FEATURES),
        "trained_on":     metadata.get("trained_on", "unknown"),
        "total_samples":  metadata.get("total_samples", "unknown"),
        "best_model":     metadata.get("best_model", "unknown"),
        "best_accuracy":  metadata.get("best_accuracy", "unknown"),
        "avg_accuracy":   metadata.get("avg_accuracy", "unknown"),
    }), 200


# ═══════════════════════════════════════════════════════════
# ROUTE 3 — /users   GET  (useful for frontend)
# ═══════════════════════════════════════════════════════════
@app.route("/users", methods=["GET"])
def users():
    return jsonify({
        "users":     list(le.classes_),
        "count":     len(le.classes_),
        "enrolled":  True
    }), 200


# ═══════════════════════════════════════════════════════════
# START SERVER
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n🚀 KeyAuth Server starting ...")
    print(f"   Health : http://localhost:5000/health")
    print(f"   Users  : http://localhost:5000/users")
    print(f"   Predict: POST http://localhost:5000/predict")
    print(f"   Ready for React app on port 3000")
    print("-" * 50)
    app.run(debug=True, host="0.0.0.0", port=5000)
