"""
BNN Inference Module for Gym Posture Analyzer
=============================================
Provides Bayesian Neural Network inference with Monte Carlo Dropout
for uncertainty estimation in exercise classification and posture scoring.

Architecture:
  - CNN branch: 3× 1D Conv layers for spatial feature extraction
  - BiLSTM branch: 2 layers for temporal sequence modeling
  - Tree-RNN branch: Recursive neural network for hierarchical body joint modeling
  - Multi-task heads: Exercise classification + Posture quality scoring
  - Bayesian: Monte Carlo Dropout for uncertainty estimation

Usage:
    from bnn_inference import BNNPredictor
    predictor = BNNPredictor("bnn_model.pt")
    result = predictor.predict(sequence_features)
"""

import json
import numpy as np
import torch
import torch.nn as nn
import joblib
from typing import Optional, Dict, List, Tuple


# ── Tree-RNN: Recursive Neural Network for Body Hierarchy ─────────────────────
class BodyTreeRNN(nn.Module):
    """
    Tree-Structured RNN modeling human body kinematic hierarchy.

    Tree structure (based on body joints):
           Root (Hip_torso)
          /    |    \
    Shoulder  Hip  Knee
       |       |     |
     Elbow   Knee  Ankle
       |
     Wrist

    Each node recursively combines child representations with parent features.
    """

    def __init__(self, input_dim=10, hidden_size=64, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_dim = input_dim

        # Joint-specific input projections (10 raw angles)
        self.joint_projections = nn.ModuleDict({
            "shoulder": nn.Linear(2, hidden_size),
            "elbow": nn.Linear(2, hidden_size),
            "hip": nn.Linear(2, hidden_size),
            "knee": nn.Linear(2, hidden_size),
            "ankle": nn.Linear(2, hidden_size),
            "wrist": nn.Linear(1, hidden_size),
        })

        # Recursive composition LSTM cells
        self.tree_rnn_cells = nn.ModuleDict({
            "root": nn.LSTMCell(hidden_size * 3, hidden_size),
            "shoulder": nn.LSTMCell(hidden_size * 2, hidden_size),
            "elbow": nn.LSTMCell(hidden_size * 2, hidden_size),
            "hip": nn.LSTMCell(hidden_size * 2, hidden_size),
            "knee": nn.LSTMCell(hidden_size * 2, hidden_size),
            "ankle": nn.LSTMCell(hidden_size, hidden_size),
            "wrist": nn.LSTMCell(hidden_size, hidden_size),
        })

        self.dropout = nn.Dropout(dropout)

    def _get_joint_features(self, x):
        """Extract joint-specific features from input."""
        joints = {
            "shoulder": x[:, [0, 5]],
            "elbow": x[:, [1, 6]],
            "hip": x[:, [2, 7]],
            "knee": x[:, [3, 8]],
            "ankle": x[:, [4, 9]],
            "wrist": x[:, 1:2],
        }
        return joints

    def forward(self, x):
        """
        Forward pass through tree hierarchy.
        Args:
            x: (batch, seq_len, input_dim) - sequence of joint angles
        Returns:
            tree_representation: (batch, hidden_size) - root node representation
        """
        batch_size = x.size(0)
        x_last = x[:, -1, :]  # Use last frame

        # Project joint inputs to hidden space
        joint_feats = self._get_joint_features(x_last)
        joint_hidden = {
            name: self.joint_projections[name](feat)
            for name, feat in joint_feats.items()
        }

        # Initialize leaf nodes
        hidden_states = {}

        # Leaf nodes
        for leaf in ["wrist", "ankle"]:
            hidden_states[leaf] = torch.tanh(self.joint_projections[leaf](joint_feats[leaf]))

        # knee_child uses knee features
        hidden_states["knee_child"] = torch.tanh(self.joint_projections["knee"](joint_feats["knee"]))

        # Process elbow (child: wrist)
        elbow_input = torch.cat([joint_hidden["elbow"], hidden_states["wrist"]], dim=1)
        h_c = self.tree_rnn_cells["elbow"](elbow_input, (
            torch.zeros(batch_size, self.hidden_size, device=x.device),
            torch.zeros(batch_size, self.hidden_size, device=x.device)
        ))
        hidden_states["elbow"] = self.dropout(h_c[0])

        # Process knee (child: ankle)
        knee_input = torch.cat([joint_hidden["knee"], hidden_states["ankle"]], dim=1)
        h_c = self.tree_rnn_cells["knee"](knee_input, (
            torch.zeros(batch_size, self.hidden_size, device=x.device),
            torch.zeros(batch_size, self.hidden_size, device=x.device)
        ))
        hidden_states["knee_child"] = self.dropout(h_c[0])

        # Process shoulder (child: elbow)
        shoulder_input = torch.cat([joint_hidden["shoulder"], hidden_states["elbow"]], dim=1)
        h_c = self.tree_rnn_cells["shoulder"](shoulder_input, (
            torch.zeros(batch_size, self.hidden_size, device=x.device),
            torch.zeros(batch_size, self.hidden_size, device=x.device)
        ))
        hidden_states["shoulder"] = self.dropout(h_c[0])

        # Process hip (child: knee_child)
        hip_input = torch.cat([joint_hidden["hip"], hidden_states["knee_child"]], dim=1)
        h_c = self.tree_rnn_cells["hip"](hip_input, (
            torch.zeros(batch_size, self.hidden_size, device=x.device),
            torch.zeros(batch_size, self.hidden_size, device=x.device)
        ))
        hidden_states["hip"] = self.dropout(h_c[0])

        # Root node: combine shoulder, hip, knee_child
        root_input = torch.cat([
            hidden_states["shoulder"],
            hidden_states["hip"],
            hidden_states["knee_child"]
        ], dim=1)
        h_c = self.tree_rnn_cells["root"](root_input, (
            torch.zeros(batch_size, self.hidden_size, device=x.device),
            torch.zeros(batch_size, self.hidden_size, device=x.device)
        ))
        tree_representation = self.dropout(h_c[0])

        return tree_representation


# ── Model Architecture (must match train_bnn_model.py) ────────────────────────
class PostureBNN(nn.Module):
    """
    CNN + BiLSTM + Tree-RNN + Multi-task + Bayesian (MC Dropout) for posture analysis.
    """
    def __init__(self, input_dim=15, cnn_hidden=64, lstm_hidden=128,
                 tree_hidden=64, fc_hidden=256, num_classes=5, dropout=0.3):
        super().__init__()

        # CNN Branch - Spatial feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_hidden, cnn_hidden * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_hidden * 2, cnn_hidden * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_hidden * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # BiLSTM Branch - Temporal sequence modeling
        self.lstm = nn.LSTM(
            input_dim=input_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_hidden > 1 else 0,
        )

        # Tree-RNN Branch - Hierarchical body joint modeling
        self.tree_rnn = BodyTreeRNN(input_dim=input_dim, hidden_size=tree_hidden, dropout=dropout)

        # Feature fusion
        cnn_feature_dim = cnn_hidden * 4 * 10  # SEQ_LEN = 10
        lstm_feature_dim = lstm_hidden * 2
        tree_feature_dim = tree_hidden

        self.fusion = nn.Sequential(
            nn.Linear(cnn_feature_dim + lstm_feature_dim + tree_feature_dim, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, fc_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Task 1: Exercise Classification
        self.classifier = nn.Linear(fc_hidden // 2, num_classes)

        # Task 2: Posture Quality Scoring
        self.posture_head = nn.Sequential(
            nn.Linear(fc_hidden // 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, return_features=False):
        batch_size = x.size(0)

        # CNN branch
        x_cnn = x.transpose(1, 2)
        x_cnn = self.cnn(x_cnn)
        x_cnn = x_cnn.transpose(1, 2)
        x_cnn_flat = x_cnn.reshape(batch_size, -1)

        # BiLSTM branch
        _, (h_n, _) = self.lstm(x)
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        x_lstm = torch.cat([h_forward, h_backward], dim=1)

        # Tree-RNN branch: hierarchical body joint modeling
        x_tree = self.tree_rnn(x)  # (batch, tree_hidden)

        # Fusion: CNN + BiLSTM + Tree-RNN
        x_fused = torch.cat([x_cnn_flat, x_lstm, x_tree], dim=1)
        x_fused = self.fusion(x_fused)

        # Task heads
        class_logits = self.classifier(x_fused)
        posture_score = self.posture_head(x_fused).squeeze(-1) * 100

        if return_features:
            return class_logits, posture_score, x_fused
        return class_logits, posture_score


# ── BNN Predictor ─────────────────────────────────────────────────────────────
class BNNPredictor:
    """
    Bayesian Neural Network predictor with Monte Carlo Dropout.

    Provides:
    - Exercise classification with confidence
    - Uncertainty estimation (σ) via MC Dropout
    - Posture quality scoring
    """

    def __init__(self, model_path: str = "bnn_model.pt",
                 scaler_path: str = "bnn_scaler.pkl",
                 encoder_path: str = "bnn_label_encoder.pkl",
                 metadata_path: str = "bnn_metadata.json",
                 mc_passes: int = 30,
                 device: Optional[str] = None):
        """
        Initialize BNN predictor.

        Args:
            model_path: Path to saved model weights (.pt)
            scaler_path: Path to StandardScaler
            encoder_path: Path to LabelEncoder
            metadata_path: Path to model metadata JSON
            mc_passes: Number of Monte Carlo passes for uncertainty
            device: 'cuda', 'cpu', or None for auto-detect
        """
        self.mc_passes = mc_passes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = None
        self.encoder = None
        self.metadata = None
        self.feature_names = None
        self.loaded = False

        # Load model and artifacts
        self._load(model_path, scaler_path, encoder_path, metadata_path)

    def _load(self, model_path: str, scaler_path: str,
              encoder_path: str, metadata_path: str):
        """Load all model artifacts."""
        try:
            # Load model weights
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model = PostureBNN().to(self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()

            # Load scaler and encoder
            self.scaler = joblib.load(scaler_path)
            self.encoder = joblib.load(encoder_path)

            # Load metadata
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)

            self.feature_names = checkpoint.get("feature_names", [])
            self.loaded = True
            print(f"BNN loaded on {self.device}")

        except FileNotFoundError as e:
            print(f"Warning: BNN model files not found: {e}")
            self.loaded = False
        except Exception as e:
            print(f"Warning: Failed to load BNN: {e}")
            self.loaded = False

    def _build_feature_vector(self, angles: Dict[str, float]) -> np.ndarray:
        """Build 15-feature vector from angle dictionary."""
        feat = {}
        for k, v in angles.items():
            feat[k] = float(v)

        # Engineered features
        feat["shoulder_elbow_ratio"] = feat.get("Shoulder_Angle", 0) / (feat.get("Elbow_Angle", 0) + 1)
        feat["hip_knee_ratio"] = feat.get("Hip_Angle", 0) / (feat.get("Knee_Angle", 0) + 1)
        feat["full_body_angle_sum"] = (feat.get("Shoulder_Angle", 0) +
                                        feat.get("Hip_Angle", 0) +
                                        feat.get("Knee_Angle", 0))
        feat["upper_body_diff"] = abs(feat.get("Shoulder_Angle", 0) - feat.get("Elbow_Angle", 0))
        feat["lower_body_diff"] = abs(feat.get("Hip_Angle", 0) - feat.get("Knee_Angle", 0))

        # Return in correct order
        return np.array([feat.get(f, 0) for f in self.feature_names])

    def _mc_dropout_inference(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Monte Carlo Dropout inference for uncertainty estimation.

        Runs self.mc_passes forward passes with dropout enabled.
        """
        self.model.train()  # Enable dropout

        all_probs = []
        all_scores = []

        with torch.no_grad():
            for _ in range(self.mc_passes):
                logits, posture = self.model(x)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())
                all_scores.append(posture.cpu().numpy())

        self.model.eval()

        probs_arr = np.stack(all_probs, axis=0)  # (mc_passes, batch, classes)
        scores_arr = np.stack(all_scores, axis=0)

        return {
            "probs_mean": np.mean(probs_arr, axis=0),
            "probs_std": np.std(probs_arr, axis=0),
            "scores_mean": np.mean(scores_arr, axis=0),
            "scores_std": np.std(scores_arr, axis=0),
        }

    def predict(self, angles: Dict[str, float],
                sequence: Optional[List[List[float]]] = None) -> Dict:
        """
        Predict exercise and posture score with uncertainty.

        Args:
            angles: Dictionary of 10 joint angles
            sequence: Optional list of historical feature vectors for temporal context

        Returns:
            Dictionary with:
            - predicted_exercise: str
            - confidence: float (0-1)
            - uncertainty_std: float (Bayesian uncertainty)
            - posture_score: float (0-100)
            - posture_uncertainty: float
            - probabilities: Dict[str, float]
        """
        if not self.loaded:
            return self._fallback_prediction(angles)

        # Build feature vector
        features = self._build_feature_vector(angles)
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        # Use sequence if provided, otherwise repeat single frame
        if sequence is not None and len(sequence) >= 10:
            seq_features = np.array(sequence[-10:])
            seq_features = self.scaler.transform(seq_features)
        else:
            seq_features = np.repeat(features_scaled.reshape(1, -1), 10, axis=0)

        # Convert to tensor
        x = torch.FloatTensor(seq_features).unsqueeze(0).to(self.device)

        # MC Dropout inference
        mc_result = self._mc_dropout_inference(x)

        probs_mean = mc_result["probs_mean"][0]
        probs_std = mc_result["probs_std"][0]
        scores_mean = mc_result["scores_mean"][0]
        scores_std = mc_result["scores_std"][0]

        # Get prediction
        pred_idx = np.argmax(probs_mean)
        confidence = float(probs_mean[pred_idx])
        uncertainty = float(np.mean(probs_std))
        exercise = self.encoder.classes_[pred_idx]
        posture_score = float(scores_mean)
        posture_uncertainty = float(scores_std)

        # Build probability dict
        probabilities = {
            cls: round(float(p), 4)
            for cls, p in zip(self.encoder.classes_, probs_mean)
        }

        return {
            "predicted_exercise": exercise,
            "confidence": round(confidence, 4),
            "uncertainty_std": round(uncertainty, 4),
            "high_uncertainty": uncertainty > 0.1,
            "posture_score": round(posture_score, 2),
            "posture_uncertainty": round(posture_uncertainty, 2),
            "posture_ci_95": [
                round(max(0, posture_score - 2 * posture_uncertainty), 2),
                round(min(100, posture_score + 2 * posture_uncertainty), 2),
            ],
            "probabilities": probabilities,
            "model": "BNN",
        }

    def _fallback_prediction(self, angles: Dict[str, float]) -> Dict:
        """Return fallback prediction when BNN is not loaded."""
        return {
            "predicted_exercise": "Unknown",
            "confidence": 0.0,
            "uncertainty_std": 0.0,
            "high_uncertainty": True,
            "posture_score": 0.0,
            "posture_uncertainty": 0.0,
            "posture_ci_95": [0, 0],
            "probabilities": {},
            "model": "BNN (not loaded)",
            "error": "BNN model not loaded",
        }

    def is_available(self) -> bool:
        """Check if BNN is loaded and ready."""
        return self.loaded


# ── Convenience Functions ─────────────────────────────────────────────────────
def load_bnn_predictor(model_dir: str = ".") -> BNNPredictor:
    """Load BNN predictor from model directory."""
    import os
    return BNNPredictor(
        model_path=os.path.join(model_dir, "bnn_model.pt"),
        scaler_path=os.path.join(model_dir, "bnn_scaler.pkl"),
        encoder_path=os.path.join(model_dir, "bnn_label_encoder.pkl"),
        metadata_path=os.path.join(model_dir, "bnn_metadata.json"),
    )


if __name__ == "__main__":
    # Demo usage
    print("Testing BNN Predictor...")
    predictor = load_bnn_predictor()

    if not predictor.is_available():
        print("BNN not loaded. Run train_bnn_model.py first.")
    else:
        # Test with sample angles
        test_angles = {
            "Shoulder_Angle": 90, "Elbow_Angle": 150, "Hip_Angle": 170,
            "Knee_Angle": 170, "Ankle_Angle": 170,
            "Shoulder_Ground_Angle": 90, "Elbow_Ground_Angle": 90,
            "Hip_Ground_Angle": 90, "Knee_Ground_Angle": 90, "Ankle_Ground_Angle": 90,
        }
        result = predictor.predict(test_angles)
        print(f"\nPrediction: {result['predicted_exercise']}")
        print(f"Confidence: {result['confidence']*100:.1f}%")
        print(f"Uncertainty: σ={result['uncertainty_std']*100:.1f}%")
        print(f"Posture Score: {result['posture_score']:.1f}/100")
