# Gym Posture Analyzer

A Flask-based web app that classifies gym exercises and scores posture in real time using a hybrid deep learning model combining CNN, BiLSTM, Tree-RNN, and Bayesian uncertainty estimation.

## Model Architecture

**Hybrid Multi-Task Deep Learning Model:**

| Component | Purpose |
|-----------|---------|
| **CNN (Convolutional Neural Network)** | 3× 1D Conv layers for spatial feature extraction from joint angles |
| **BiLSTM (Bidirectional LSTM)** | 2-layer recurrent network for temporal sequence modeling |
| **Tree-RNN (Recursive Neural Network)** | Tree-structured RNN modeling hierarchical body joint relationships |
| **Multi-task Learning** | Dual heads: exercise classification + posture quality scoring |
| **Bayesian (MC Dropout)** | Monte Carlo Dropout for uncertainty estimation |

**Tree-RNN Structure (models human body kinematic chain):**
```
       Root (Torso/Hip)
      /    |    \
  Shoulder  Hip  Knee
    |       |     |
  Elbow   Knee  Ankle
    |
  Wrist
```

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000

## API

### POST /api/analyze
```json
{
  "angles": {
    "Shoulder_Angle": 90, "Elbow_Angle": 150, "Hip_Angle": 170,
    "Knee_Angle": 170, "Ankle_Angle": 170,
    "Shoulder_Ground_Angle": 90, "Elbow_Ground_Angle": 90,
    "Hip_Ground_Angle": 90, "Knee_Ground_Angle": 90, "Ankle_Ground_Angle": 90
  },
  "exercise": null
}
```

### GET /api/metadata
Returns model accuracy, feature importances, confusion matrix.

## Model
- Random Forest, 200 trees, max_depth=15
- 15 features (10 raw + 5 engineered)
- 5 classes: Jumping Jacks, Pull ups, Push Ups, Russian twists, Squats
- 31,033 training samples
