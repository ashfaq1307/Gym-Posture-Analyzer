"""
Gym Posture Analyzer — Flask Backend
=====================================
Endpoints:
  POST /api/analyze   — predict exercise + posture score (Random Forest)
  POST /api/analyze_advanced — predict with BNN + uncertainty
  GET  /api/metadata  — return model metadata
  GET  /api/validate_accuracy — run live validation and return accuracy metrics
  GET  /api/load_bnn  — lazy-load BNN model
  GET  /              — serve frontend
"""

import json
import warnings
import numpy as np
import joblib
from flask import Flask, request, jsonify, send_from_directory
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")

app = Flask(__name__, static_folder="static")

# ── Load model artifacts ─────────────────────────────────────────────────────
MODEL  = joblib.load("posture_model.pkl")
LE     = joblib.load("label_encoder.pkl")
FEATS  = joblib.load("feature_names.pkl")
META   = json.load(open("model_metadata.json"))

# ── BNN Predictor (lazy-loaded) ──────────────────────────────────────────────
BNN_PREDICTOR = None
BNN_AVAILABLE = False

def get_bnn_predictor():
    """Lazy-load BNN predictor on first request."""
    global BNN_PREDICTOR, BNN_AVAILABLE
    if BNN_PREDICTOR is None:
        try:
            from bnn_inference import BNNPredictor
            BNN_PREDICTOR = BNNPredictor()
            BNN_AVAILABLE = BNN_PREDICTOR.is_available()
            if BNN_AVAILABLE:
                print("[BNN] Loaded successfully")
            else:
                print("[BNN] Model files not found - run train_bnn_model.py first")
        except Exception as e:
            print(f"[BNN] Failed to load: {e}")
            BNN_AVAILABLE = False
    return BNN_PREDICTOR

RAW_ANGLE_COLS = [
    "Shoulder_Angle", "Elbow_Angle", "Hip_Angle", "Knee_Angle", "Ankle_Angle",
    "Shoulder_Ground_Angle", "Elbow_Ground_Angle", "Hip_Ground_Angle",
    "Knee_Ground_Angle", "Ankle_Ground_Angle",
]

# ── Biomechanical posture rules ───────────────────────────────────────────────
POSTURE_RULES = {
    "Squats": [
        {"joint": "Knee_Angle",     "min": 70,  "max": 160, "weight": 3, "tip": "Knee tracks over toe — 70°–160° squat range", "phase": "Descent"},
        {"joint": "Hip_Angle",      "min": 70,  "max": 160, "weight": 3, "tip": "Hip hinge — avoid collapsing forward below 70°", "phase": "Hinge"},
        {"joint": "Shoulder_Angle", "min": 0,   "max": 90,  "weight": 1, "tip": "Upright torso — shoulders back", "phase": "Posture"},
        {"joint": "Ankle_Angle",    "min": 140, "max": 180, "weight": 2, "tip": "Dorsiflexion — heels flat on ground", "phase": "Stability"},
    ],
    "Push Ups": [
        {"joint": "Hip_Angle",      "min": 165, "max": 180, "weight": 3, "tip": "Plank line — no sagging or piking hips", "phase": "Plank"},
        {"joint": "Elbow_Angle",    "min": 80,  "max": 170, "weight": 2, "tip": "Full ROM — chest near ground at bottom", "phase": "Descent"},
        {"joint": "Shoulder_Angle", "min": 45,  "max": 90,  "weight": 2, "tip": "Shoulders over wrists — elbows at ~45°", "phase": "Alignment"},
        {"joint": "Knee_Angle",     "min": 165, "max": 180, "weight": 2, "tip": "Legs straight — no bent knees", "phase": "Plank"},
    ],
    "Pull ups": [
        {"joint": "Shoulder_Angle", "min": 80,  "max": 165, "weight": 3, "tip": "Full shoulder extension at bottom", "phase": "Full ROM"},
        {"joint": "Elbow_Angle",    "min": 70,  "max": 170, "weight": 3, "tip": "Full elbow flex at top — chin over bar", "phase": "Pull"},
        {"joint": "Hip_Angle",      "min": 160, "max": 180, "weight": 2, "tip": "Hollow body — slight anterior tilt", "phase": "Core"},
    ],
    "Jumping Jacks": [
        {"joint": "Shoulder_Angle", "min": 10,  "max": 175, "weight": 2, "tip": "Arms sweep full arc — 10° to 175° overhead", "phase": "Arms"},
        {"joint": "Knee_Angle",     "min": 165, "max": 180, "weight": 2, "tip": "Knees mostly straight throughout", "phase": "Legs"},
        {"joint": "Ankle_Angle",    "min": 140, "max": 180, "weight": 1, "tip": "Land softly — ankle absorbs impact", "phase": "Landing"},
    ],
    "Russian twists": [
        {"joint": "Hip_Angle",      "min": 90,  "max": 130, "weight": 3, "tip": "V-sit position — hips at 90°–130°", "phase": "Core"},
        {"joint": "Knee_Angle",     "min": 120, "max": 160, "weight": 2, "tip": "Knees bent, feet off floor or lightly touching", "phase": "Legs"},
        {"joint": "Shoulder_Angle", "min": 30,  "max": 80,  "weight": 2, "tip": "Arms extended — rotate fully side to side", "phase": "Rotation"},
    ],
}


def build_features(angles: dict) -> np.ndarray:
    """Build the 15-feature vector the model expects."""
    feat = {k: float(angles.get(k, 0.0)) for k in RAW_ANGLE_COLS}
    feat["shoulder_elbow_ratio"] = feat["Shoulder_Angle"] / (feat["Elbow_Angle"] + 1)
    feat["hip_knee_ratio"]       = feat["Hip_Angle"] / (feat["Knee_Angle"] + 1)
    feat["full_body_angle_sum"]  = feat["Shoulder_Angle"] + feat["Hip_Angle"] + feat["Knee_Angle"]
    feat["upper_body_diff"]      = abs(feat["Shoulder_Angle"] - feat["Elbow_Angle"])
    feat["lower_body_diff"]      = abs(feat["Hip_Angle"] - feat["Knee_Angle"])
    return np.array([[feat[f] for f in FEATS]])


def score_posture(angles: dict, exercise: str) -> dict:
    """Score posture 0–100 and return per-joint checks."""
    rules = POSTURE_RULES.get(exercise, [])
    if not rules:
        return {"score": 75, "checks": [], "passed": 0, "total": 0}

    total_w = sum(r["weight"] for r in rules)
    ok_w = 0
    checks = []

    for r in rules:
        val = float(angles.get(r["joint"], 0))
        in_range = (r.get("min", 0) <= val <= r.get("max", 180))
        if in_range:
            ok_w += r["weight"]
        deviation = 0
        if not in_range:
            lo_dev = abs(val - r["min"]) if r.get("min") is not None else 999
            hi_dev = abs(val - r["max"]) if r.get("max") is not None else 999
            deviation = min(lo_dev, hi_dev)
        checks.append({
            "joint":   r["joint"],
            "value":   round(val, 1),
            "ok":      in_range,
            "min":     r.get("min"),
            "max":     r.get("max"),
            "weight":  r["weight"],
            "tip":     r["tip"],
            "phase":   r["phase"],
            "deviation": round(deviation, 1),
        })

    score = round((ok_w / total_w) * 100) if total_w else 75
    passed = sum(1 for c in checks if c["ok"])
    rating = "GOOD" if score >= 75 else "MEDIUM" if score >= 50 else "BAD"
    return {"score": score, "rating": rating, "checks": checks, "passed": passed, "total": len(checks)}


# ── API Routes ────────────────────────────────────────────────────────────────

@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)
    angles   = data.get("angles", {})
    override = data.get("exercise")          # optional user override

    # Inference
    X = build_features(angles)
    pred_idx  = MODEL.predict(X)[0]
    probs_arr = MODEL.predict_proba(X)[0]
    predicted = LE.classes_[pred_idx]
    confidence = float(probs_arr[pred_idx])

    exercise = override if override in POSTURE_RULES else predicted

    # Posture
    posture = score_posture(angles, exercise)

    return jsonify({
        "predicted_exercise": predicted,
        "active_exercise":    exercise,
        "confidence":         round(confidence, 4),
        "probabilities": {
            cls: round(float(p), 4)
            for cls, p in zip(LE.classes_, probs_arr)
        },
        "posture": posture,
    })


@app.route("/api/analyze_advanced", methods=["POST"])
def analyze_advanced():
    """Advanced endpoint with BNN + CNN + BiLSTM + Multi-task inference."""
    data = request.get_json(force=True)
    angles   = data.get("angles", {})
    override = data.get("exercise")
    sequence = data.get("sequence", [])

    # Get BNN predictor
    bnn = get_bnn_predictor()

    if BNN_AVAILABLE and bnn is not None:
        # Use real BNN inference
        result = bnn.predict(angles, sequence)

        # Map BNN result to response format
        exercise = override if override in POSTURE_RULES else result["predicted_exercise"]

        # Also compute RF-based posture rules for joint feedback
        posture_rules = score_posture(angles, exercise)

        return jsonify({
            "predicted_exercise": result["predicted_exercise"],
            "active_exercise":    exercise,
            "confidence":         result["confidence"],
            "probabilities":      result["probabilities"],
            "posture": {
                "score": result["posture_score"],
                "rating": "GOOD" if result["posture_score"] >= 75 else "MEDIUM" if result["posture_score"] >= 50 else "BAD",
                "checks": posture_rules["checks"],
                "passed": posture_rules["passed"],
                "total": posture_rules["total"],
            },
            "model": "BNN",
            "bnn": {
                "confidence": result["confidence"],
                "uncertainty_std": result["uncertainty_std"],
                "high_uncertainty": result["high_uncertainty"],
                "posture_score_nn": result["posture_score"],
                "posture_uncertainty": result["posture_uncertainty"],
                "posture_ci_95": result["posture_ci_95"],
            },
        })

    else:
        # Fallback to Random Forest
        X = build_features(angles)
        pred_idx  = MODEL.predict(X)[0]
        probs_arr = MODEL.predict_proba(X)[0]
        predicted = LE.classes_[pred_idx]
        confidence = float(probs_arr[pred_idx])

        exercise = override if override in POSTURE_RULES else predicted
        posture = score_posture(angles, exercise)

        return jsonify({
            "predicted_exercise": predicted,
            "active_exercise":    exercise,
            "confidence":         round(confidence, 4),
            "probabilities": {
                cls: round(float(p), 4)
                for cls, p in zip(LE.classes_, probs_arr)
            },
            "posture": posture,
            "model": "RF",
            "bnn": None,
        })


@app.route("/api/metadata", methods=["GET"])
def metadata():
    # Update with BNN availability
    META["bnn_available"] = BNN_AVAILABLE
    if BNN_AVAILABLE:
        try:
            with open("bnn_metadata.json", "r") as f:
                META["bnn_model"] = json.load(f)
        except:
            META["bnn_model"] = {}
    return jsonify(META)


@app.route("/api/validate_accuracy", methods=["GET"])
def validate_accuracy():
    """
    Run live validation on the loaded model and return accuracy metrics.
    This endpoint validates the model against test data and prints results to console.
    """
    print("\n" + "=" * 60)
    print("  MODEL ACCURACY VALIDATION — LIVE TEST")
    print("=" * 60)

    # Load test data from exercise_angles.csv
    try:
        import pandas as pd
        df = pd.read_csv("exercise_angles.csv")

        # Separate features and labels
        label_col = "Exercise"  # assuming this is the label column
        feature_cols = FEATS  # use the same 15 features

        X_test = df[feature_cols].values
        y_test = df[label_col].values

        # Encode labels
        y_test_encoded = LE.transform(y_test)

        # Run prediction
        y_pred = MODEL.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test_encoded, y_pred)
        cm = confusion_matrix(y_test_encoded, y_pred)
        class_report = classification_report(y_test_encoded, y_pred, target_names=LE.classes_)

        # Per-class accuracy
        per_class_acc = cm.diagonal() / cm.sum(axis=1)

        # Print to console
        print(f"\n  Test Samples: {len(y_test)}")
        print(f"  Overall Accuracy: {accuracy * 100:.2f}%")
        print(f"\n  Per-Class Accuracy:")
        for i, cls in enumerate(LE.classes_):
            print(f"    {cls}: {per_class_acc[i] * 100:.2f}% ({cm[i].sum()} samples)")
        print(f"\n  Confusion Matrix:")
        print(cm)
        print(f"\n  Classification Report:")
        print(class_report)
        print("=" * 60 + "\n")

        return jsonify({
            "status": "success",
            "test_samples": int(len(y_test)),
            "overall_accuracy": float(accuracy),
            "per_class_accuracy": {cls: float(per_class_acc[i]) for i, cls in enumerate(LE.classes_)},
            "confusion_matrix": cm.tolist(),
            "classification_report": class_report
        })

    except Exception as e:
        print(f"[VALIDATION ERROR] {e}")
        # Fallback: return stored metrics
        return jsonify({
            "status": "fallback",
            "message": str(e),
            "stored_accuracy": META.get("accuracy", 0),
            "stored_cv_mean": META.get("cv_mean", 0),
            "stored_cv_std": META.get("cv_std", 0)
        })


@app.route("/api/load_bnn", methods=["GET"])
def load_bnn():
    """Endpoint to manually trigger BNN loading."""
    bnn = get_bnn_predictor()
    if BNN_AVAILABLE:
        return jsonify({"status": "loaded", "message": "BNN ready"})
    else:
        return jsonify({"status": "error", "message": "BNN not found - run train_bnn_model.py first"}), 404


@app.route("/", methods=["GET"])
def index():
    return send_from_directory("static", "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("static", path)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("       GYM POSTURE ANALYZER — Model Validation Report")
    print("=" * 60)
    print(f"\n  Model Type: {META['model_type']}")
    print(f"\n  ACCURACY METRICS:")
    print(f"    • Test Accuracy:      {META['accuracy'] * 100:.2f}%")
    print(f"    • CV Mean (5-fold):   {META['cv_mean'] * 100:.2f}% (+/- {META['cv_std'] * 100:.2f}%)")
    print(f"\n  Dataset:")
    print(f"    • Training samples:   {META.get('train_samples', 'N/A'):,}")
    print(f"    • Test samples:       {META.get('test_samples', 'N/A'):,}")
    print(f"\n  Classes ({len(META['classes'])}):")
    for i, cls in enumerate(META['classes']):
        print(f"    {i + 1}. {cls}")
    print(f"\n  Feature Importances (Top 5):")
    feat_imp = META.get('feature_importances', {})
    sorted_imp = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:5]
    for feat, imp in sorted_imp:
        print(f"    • {feat}: {imp * 100:.2f}%")
    print("\n" + "=" * 60)
    print(f"  Starting server at http://localhost:5000")
    print(f"  API Docs: GET /api/metadata | GET /api/validate_accuracy")
    print("=" * 60 + "\n")
    app.run(debug=True, port=5000)