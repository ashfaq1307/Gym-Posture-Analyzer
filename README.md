# Gym Posture Analyzer

A Flask web application that classifies gym exercises and scores posture quality from joint angle data. Uses a Random Forest classifier (95.76% accuracy) as the primary model, with an optional advanced Bayesian Neural Network (BNN) combining CNN, BiLSTM, and Tree-RNN for uncertainty-aware predictions.

---

## Features

- **Exercise classification** across 5 exercises: Jumping Jacks, Pull-ups, Push Ups, Russian Twists, Squats
- **Posture scoring** (0–100) using per-exercise biomechanical joint angle rules
- **Dual inference modes** — fast Random Forest or advanced BNN with uncertainty estimation
- **Per-joint feedback** with phase labels, allowed ranges, and corrective tips
- **REST API** for integration with webcam or sensor pipelines
- **Live accuracy validation** endpoint for on-demand model evaluation

---

## Model Architecture

### Primary — Random Forest

| Parameter | Value |
|-----------|-------|
| Trees | 200 |
| Max depth | 15 |
| Test accuracy | **95.76%** |
| 5-fold CV mean | **96.25% ± 0.11%** |
| Training samples | 24,826 |
| Test samples | 6,207 |
| Features | 15 (10 raw + 5 engineered) |

**Top 5 features by importance:**

| Feature | Importance |
|---------|-----------|
| Ankle_Angle | 23.21% |
| full_body_angle_sum | 10.48% |
| hip_knee_ratio | 10.20% |
| Hip_Angle | 9.29% |
| upper_body_diff | 8.92% |

### Advanced — BNN (optional)

A multi-branch deep learning model with Bayesian uncertainty estimation via Monte Carlo Dropout (30 passes).

| Component | Details |
|-----------|---------|
| **CNN branch** | 3× 1D Conv layers (64→128→256 channels) with BatchNorm + ReLU for spatial feature extraction |
| **BiLSTM branch** | 2-layer bidirectional LSTM (hidden=128) for temporal sequence modeling over 10-frame windows |
| **Tree-RNN branch** | Recursive LSTM cells modeling the human body kinematic chain (hip→shoulder→elbow→wrist, hip→knee→ankle) |
| **Fusion** | Concatenated CNN + BiLSTM + Tree-RNN features → FC(256) → FC(128) |
| **Task 1** | Exercise classification head (5 classes, CrossEntropy) |
| **Task 2** | Posture quality regression head (0–100, Sigmoid × 100) |
| **Bayesian** | MC Dropout (p=0.3, 30 forward passes) → mean prediction + σ uncertainty |

**Tree-RNN body hierarchy:**
```
       Root (Hip/Torso)
      /      |      \
 Shoulder   Hip    Knee
    |         |       |
  Elbow     Knee   Ankle
    |
  Wrist
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app (Random Forest loads automatically)
python app.py

# 3. Open browser
# http://localhost:5000
```

To also enable the BNN model, train it first:

```bash
python train_bnn_model.py --data exercise_angles.csv --epochs 50
```

This saves `bnn_model.pt`, `bnn_scaler.pkl`, `bnn_label_encoder.pkl`, and `bnn_metadata.json`. The BNN is then lazy-loaded on the first call to `/api/analyze_advanced`.

---

## API Reference

### `POST /api/analyze`
Classify exercise and score posture using the Random Forest model.

**Request:**
```json
{
  "angles": {
    "Shoulder_Angle": 90,
    "Elbow_Angle": 150,
    "Hip_Angle": 170,
    "Knee_Angle": 170,
    "Ankle_Angle": 170,
    "Shoulder_Ground_Angle": 90,
    "Elbow_Ground_Angle": 90,
    "Hip_Ground_Angle": 90,
    "Knee_Ground_Angle": 90,
    "Ankle_Ground_Angle": 90
  },
  "exercise": null
}
```
Set `"exercise"` to a class name (e.g. `"Squats"`) to override the prediction and use its posture rules.

**Response:**
```json
{
  "predicted_exercise": "Squats",
  "active_exercise": "Squats",
  "confidence": 0.9812,
  "probabilities": { "Squats": 0.9812, "Push Ups": 0.011, "...": "..." },
  "posture": {
    "score": 75,
    "rating": "GOOD",
    "passed": 3,
    "total": 4,
    "checks": [
      {
        "joint": "Knee_Angle", "value": 170.0, "ok": true,
        "min": 70, "max": 160, "weight": 3,
        "tip": "Knee tracks over toe — 70°–160° squat range",
        "phase": "Descent", "deviation": 0
      }
    ]
  }
}
```

---

### `POST /api/analyze_advanced`
BNN inference with uncertainty. Falls back to Random Forest if BNN is not trained.

**Request:** Same as `/api/analyze`, plus optional `"sequence"` (list of up to 10 prior feature vectors for temporal context).

**Additional response fields (BNN mode):**
```json
{
  "model": "BNN",
  "bnn": {
    "confidence": 0.94,
    "uncertainty_std": 0.032,
    "high_uncertainty": false,
    "posture_score_nn": 81.5,
    "posture_uncertainty": 3.2,
    "posture_ci_95": [75.1, 87.9]
  }
}
```

---

### `GET /api/metadata`
Returns model type, accuracy, CV scores, confusion matrix, feature importances, and class list.

### `GET /api/validate_accuracy`
Runs live validation against `exercise_angles.csv` and returns per-class accuracy, confusion matrix, and classification report.

### `GET /api/load_bnn`
Manually triggers BNN model loading without making a prediction.

---

## Features (Input)

The model expects 10 raw joint angles. Five additional engineered features are computed automatically:

| Feature | Description |
|---------|-------------|
| `Shoulder_Angle` | Shoulder joint angle (degrees) |
| `Elbow_Angle` | Elbow joint angle |
| `Hip_Angle` | Hip joint angle |
| `Knee_Angle` | Knee joint angle |
| `Ankle_Angle` | Ankle joint angle |
| `*_Ground_Angle` | Angle of each joint relative to ground plane |
| `shoulder_elbow_ratio` | Shoulder / (Elbow + 1) |
| `hip_knee_ratio` | Hip / (Knee + 1) |
| `full_body_angle_sum` | Shoulder + Hip + Knee |
| `upper_body_diff` | \|Shoulder − Elbow\| |
| `lower_body_diff` | \|Hip − Knee\| |

---

## Dataset

`exercise_angles.csv` — 31,033 samples, 10 angle columns + `Exercise` label.

| Exercise | Approx. samples |
|----------|----------------|
| Jumping Jacks | ~6,200 |
| Pull ups | ~6,600 |
| Push Ups | ~9,800 |
| Russian twists | ~4,400 |
| Squats | ~4,900 |

---

## Project Structure

```
gym-posture-analyzer/
├── app.py                  # Flask backend and API routes
├── bnn_inference.py        # BNN model definition and MC Dropout predictor
├── train_bnn_model.py      # BNN training script
├── requirements.txt        # Python dependencies
├── README.md
├── model_metadata.json     # RF accuracy stats and feature importances
├── exercise_angles.csv     # Training dataset (31K samples)
├── posture_model.pkl       # Trained Random Forest (35 MB)
├── feature_names.pkl       # Feature name order for inference
├── label_encoder.pkl       # Exercise class encoder
└── static/
    └── index.html          # Web frontend
```

After training the BNN, these files are also generated:
```
├── bnn_model.pt            # Trained BNN weights (PyTorch)
├── bnn_scaler.pkl          # StandardScaler for BNN features
├── bnn_label_encoder.pkl   # LabelEncoder for BNN classes
└── bnn_metadata.json       # BNN accuracy and confusion matrix
```

---

## Requirements

```
flask>=3.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
joblib>=1.3.0
torch>=2.0.0
torchvision>=0.15.0
pandas>=2.0.0
```

GPU is supported for BNN training/inference. If CUDA is unavailable the model runs on CPU automatically.

---

## Posture Rating

| Score | Rating |
|-------|--------|
| 75–100 | GOOD |
| 50–74 | MEDIUM |
| 0–49 | BAD |

Scores are computed by checking each joint angle against exercise-specific biomechanical ranges. Each joint check is weighted by its importance (e.g. knee tracking in squats has weight 3, ankle stability has weight 2).
