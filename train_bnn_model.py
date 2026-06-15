"""
Train BNN + CNN + BiLSTM + Tree-RNN + Multi-task Model for Gym Posture Analysis
================================================================================
Architecture:
  - CNN branch: 3× 1D Conv layers for spatial feature extraction
  - BiLSTM branch: 2 layers for temporal sequence modeling
  - Tree-RNN branch: Recursive neural network for hierarchical body joint modeling
  - Multi-task heads: Exercise classification + Posture quality scoring
  - Bayesian: Monte Carlo Dropout for uncertainty estimation

Tree-RNN Structure (models human body kinematic chain):
       Root (Torso/Hip)
      /    |    \
  Shoulder  Hip  Knee
    |       |     |
  Elbow   Knee  Ankle
    |
  Wrist

Usage:
  python train_bnn_model.py --data exercise_angles.csv --epochs 50
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


# ── Configuration ─────────────────────────────────────────────────────────────
SEQ_LEN = 10
INPUT_DIM = 15  # 10 raw angles + 5 engineered features
CNN_HIDDEN = 64
LSTM_HIDDEN = 128
TREE_HIDDEN = 64  # Tree-RNN hidden size
FC_HIDDEN = 256
NUM_CLASSES = 5
DROPOUT = 0.3
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3


# ── Dataset ───────────────────────────────────────────────────────────────────
class PostureDataset(Dataset):
    def __init__(self, sequences, labels, posture_scores=None):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        self.posture_scores = torch.FloatTensor(posture_scores) if posture_scores is not None else None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.posture_scores is not None:
            return self.sequences[idx], self.labels[idx], self.posture_scores[idx]
        return self.sequences[idx], self.labels[idx]


def build_features(df):
    """Build 15 features from DataFrame with 10 angle columns."""
    features = df.copy()
    features["shoulder_elbow_ratio"] = features["Shoulder_Angle"] / (features["Elbow_Angle"] + 1)
    features["hip_knee_ratio"] = features["Hip_Angle"] / (features["Knee_Angle"] + 1)
    features["full_body_angle_sum"] = features["Shoulder_Angle"] + features["Hip_Angle"] + features["Knee_Angle"]
    features["upper_body_diff"] = np.abs(features["Shoulder_Angle"] - features["Elbow_Angle"])
    features["lower_body_diff"] = np.abs(features["Hip_Angle"] - features["Knee_Angle"])
    return features


def create_sequences(features, labels, seq_len=SEQ_LEN):
    """Create overlapping sequences for temporal modeling."""
    X, y = [], []
    for i in range(len(features) - seq_len + 1):
        X.append(features[i:i + seq_len])
        y.append(labels[i + seq_len - 1])  # Predict last frame's label
    return np.array(X), np.array(y)


def estimate_posture_score(angles_dict, exercise):
    """
    Estimate posture quality (0-100) based on biomechanical rules.
    This creates pseudo-labels for multi-task training.
    """
    RULES = {
        "Squats": [
            {"joint": "Knee_Angle", "min": 70, "max": 160, "weight": 3},
            {"joint": "Hip_Angle", "min": 70, "max": 160, "weight": 3},
            {"joint": "Shoulder_Angle", "min": 0, "max": 90, "weight": 1},
            {"joint": "Ankle_Angle", "min": 140, "max": 180, "weight": 2},
        ],
        "Push Ups": [
            {"joint": "Hip_Angle", "min": 165, "max": 180, "weight": 3},
            {"joint": "Elbow_Angle", "min": 80, "max": 170, "weight": 2},
            {"joint": "Shoulder_Angle", "min": 45, "max": 90, "weight": 2},
            {"joint": "Knee_Angle", "min": 165, "max": 180, "weight": 2},
        ],
        "Pull ups": [
            {"joint": "Shoulder_Angle", "min": 80, "max": 165, "weight": 3},
            {"joint": "Elbow_Angle", "min": 70, "max": 170, "weight": 3},
            {"joint": "Hip_Angle", "min": 160, "max": 180, "weight": 2},
        ],
        "Jumping Jacks": [
            {"joint": "Shoulder_Angle", "min": 10, "max": 175, "weight": 2},
            {"joint": "Knee_Angle", "min": 165, "max": 180, "weight": 2},
            {"joint": "Ankle_Angle", "min": 140, "max": 180, "weight": 1},
        ],
        "Russian twists": [
            {"joint": "Hip_Angle", "min": 90, "max": 130, "weight": 3},
            {"joint": "Knee_Angle", "min": 120, "max": 160, "weight": 2},
            {"joint": "Shoulder_Angle", "min": 30, "max": 80, "weight": 2},
        ],
    }

    rules = RULES.get(exercise, [])
    if not rules:
        return 75.0

    total_w = sum(r["weight"] for r in rules)
    ok_w = 0
    for r in rules:
        val = float(angles_dict.get(r["joint"], 0))
        if r["min"] <= val <= r["max"]:
            ok_w += r["weight"]
    return round((ok_w / total_w) * 100, 1)


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

    # Define body hierarchy: parent -> [children]
    BODY_TREE = {
        "root": ["shoulder", "hip", "knee"],
        "shoulder": ["elbow"],
        "elbow": ["wrist"],
        "hip": ["knee_child"],  # hip_knee
        "knee": ["ankle"],
        "knee_child": [],
        "wrist": [],
        "ankle": [],
    }

    def __init__(self, input_dim=10, hidden_size=TREE_HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_dim = input_dim

        # Joint-specific input projections (10 raw angles)
        self.joint_projections = nn.ModuleDict({
            "shoulder": nn.Linear(2, hidden_size),  # Shoulder + Shoulder_Ground
            "elbow": nn.Linear(2, hidden_size),    # Elbow + Elbow_Ground
            "hip": nn.Linear(2, hidden_size),      # Hip + Hip_Ground
            "knee": nn.Linear(2, hidden_size),     # Knee + Knee_Ground
            "ankle": nn.Linear(2, hidden_size),    # Ankle + Ankle_Ground
            "wrist": nn.Linear(1, hidden_size),    # Derived feature
        })

        # Recursive composition: combine parent + children representations
        self.tree_rnn_cells = nn.ModuleDict({
            "root": nn.LSTMCell(hidden_size * 3, hidden_size),
            "shoulder": nn.LSTMCell(hidden_size * 2, hidden_size),
            "elbow": nn.LSTMCell(hidden_size * 2, hidden_size),
            "hip": nn.LSTMCell(hidden_size * 2, hidden_size),
            "knee": nn.LSTMCell(hidden_size * 2, hidden_size),
            "knee_child": nn.LSTMCell(hidden_size, hidden_size),
            "ankle": nn.LSTMCell(hidden_size, hidden_size),
            "wrist": nn.LSTMCell(hidden_size, hidden_size),
        })

        self.dropout = nn.Dropout(dropout)

    def _get_joint_features(self, x):
        """
        Extract joint-specific features from input.
        Args:
            x: (batch, input_dim) - 10 raw angles + 5 engineered
        Returns:
            dict of joint feature tensors
        """
        # x columns: [Shoulder, Elbow, Hip, Knee, Ankle, Shoulder_G, Elbow_G, Hip_G, Knee_G, Ankle_G, ...]
        joints = {
            "shoulder": x[:, [0, 5]],    # Shoulder_Angle, Shoulder_Ground_Angle
            "elbow": x[:, [1, 6]],       # Elbow_Angle, Elbow_Ground_Angle
            "hip": x[:, [2, 7]],         # Hip_Angle, Hip_Ground_Angle
            "knee": x[:, [3, 8]],        # Knee_Angle, Knee_Ground_Angle
            "ankle": x[:, [4, 9]],       # Ankle_Angle, Ankle_Ground_Angle
            "wrist": x[:, [1:2]],        # Use elbow as proxy for wrist
        }
        return {k: v for k, v in joints.items()}

    def forward(self, x):
        """
        Forward pass through tree hierarchy.
        Args:
            x: (batch, seq_len, input_dim) - sequence of joint angles
        Returns:
            tree_representation: (batch, hidden_size) - root node representation
        """
        batch_size = x.size(0)
        # Use last frame for tree structure
        x_last = x[:, -1, :]  # (batch, input_dim)

        # Project joint inputs to hidden space
        joint_feats = self._get_joint_features(x_last)
        joint_hidden = {
            name: self.joint_projections[name](feat)
            for name, feat in joint_feats.items()
        }

        # Initialize leaf nodes (no children)
        leaf_nodes = ["wrist", "ankle", "knee_child"]
        hidden_states = {}
        cell_states = {}

        # Leaf nodes: just use their projection
        for leaf in leaf_nodes:
            if leaf == "knee_child":
                hidden_states[leaf] = torch.tanh(self.joint_projections["knee"](joint_feats["knee"]))
            else:
                hidden_states[leaf] = torch.tanh(self.joint_projections[leaf](joint_feats[leaf]))
            cell_states[leaf] = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # Process elbow (child: wrist)
        elbow_input = torch.cat([joint_hidden["elbow"], hidden_states["wrist"]], dim=1)
        h_c = self.tree_rnn_cells["elbow"](elbow_input, (
            torch.zeros(batch_size, self.hidden_size, device=x.device),
            torch.zeros(batch_size, self.hidden_size, device=x.device)
        ))
        hidden_states["elbow"] = self.dropout(h_c[0])
        cell_states["elbow"] = h_c[1]

        # Process knee_child -> knee (child: ankle)
        knee_child_input = torch.cat([joint_hidden["knee"], hidden_states["ankle"]], dim=1)
        h_c = self.tree_rnn_cells["knee"](knee_child_input, (
            torch.zeros(batch_size, self.hidden_size, device=x.device),
            torch.zeros(batch_size, self.hidden_size, device=x.device)
        ))
        hidden_states["knee_child"] = self.dropout(h_c[0])
        cell_states["knee_child"] = h_c[1]

        # Process shoulder (child: elbow)
        shoulder_input = torch.cat([joint_hidden["shoulder"], hidden_states["elbow"]], dim=1)
        h_c = self.tree_rnn_cells["shoulder"](shoulder_input, (
            torch.zeros(batch_size, self.hidden_size, device=x.device),
            torch.zeros(batch_size, self.hidden_size, device=x.device)
        ))
        hidden_states["shoulder"] = self.dropout(h_c[0])
        cell_states["shoulder"] = h_c[1]

        # Process hip (child: knee_child)
        hip_input = torch.cat([joint_hidden["hip"], hidden_states["knee_child"]], dim=1)
        h_c = self.tree_rnn_cells["hip"](hip_input, (
            torch.zeros(batch_size, self.hidden_size, device=x.device),
            torch.zeros(batch_size, self.hidden_size, device=x.device)
        ))
        hidden_states["hip"] = self.dropout(h_c[0])
        cell_states["hip"] = h_c[1]

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
        tree_representation = self.dropout(h_c[0])  # (batch, hidden_size)

        return tree_representation


# ── Model Architecture ────────────────────────────────────────────────────────
class PostureBNN(nn.Module):
    """
    CNN + BiLSTM + Tree-RNN + Multi-task + Bayesian (MC Dropout) for posture analysis.
    """
    def __init__(self, input_dim=INPUT_DIM, cnn_hidden=CNN_HIDDEN,
                 lstm_hidden=LSTM_HIDDEN, tree_hidden=TREE_HIDDEN,
                 fc_hidden=FC_HIDDEN, num_classes=NUM_CLASSES, dropout=DROPOUT):
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
        cnn_feature_dim = cnn_hidden * 4 * SEQ_LEN  # Flatten CNN output
        lstm_feature_dim = lstm_hidden * 2  # Bidirectional final state
        tree_feature_dim = tree_hidden  # Tree-RNN root representation

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

        # Task 2: Posture Quality Scoring (regression 0-100)
        self.posture_head = nn.Sequential(
            nn.Linear(fc_hidden // 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Output 0-1, scaled to 0-100
        )

    def forward(self, x, return_features=False):
        """
        Args:
            x: (batch, seq_len, input_dim)
            return_features: if True, return fused features for uncertainty estimation
        Returns:
            class_logits, posture_score, features (optional)
        """
        batch_size = x.size(0)

        # CNN branch: (batch, seq_len, input_dim) -> (batch, seq_len, cnn_hidden*4)
        x_cnn = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        x_cnn = self.cnn(x_cnn)    # (batch, cnn_hidden*4, seq_len)
        x_cnn = x_cnn.transpose(1, 2)  # (batch, seq_len, cnn_hidden*4)
        x_cnn_flat = x_cnn.reshape(batch_size, -1)  # Flatten for fusion

        # BiLSTM branch
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers * 2, batch, lstm_hidden)
        h_forward = h_n[-2, :, :]   # Last layer forward
        h_backward = h_n[-1, :, :]  # Last layer backward
        x_lstm = torch.cat([h_forward, h_backward], dim=1)  # (batch, lstm_hidden*2)

        # Tree-RNN branch: hierarchical body joint modeling
        x_tree = self.tree_rnn(x)  # (batch, tree_hidden)

        # Fusion: CNN + BiLSTM + Tree-RNN
        x_fused = torch.cat([x_cnn_flat, x_lstm, x_tree], dim=1)
        x_fused = self.fusion(x_fused)

        # Task heads
        class_logits = self.classifier(x_fused)
        posture_score = self.posture_head(x_fused).squeeze(-1) * 100  # Scale to 0-100

        if return_features:
            return class_logits, posture_score, x_fused
        return class_logits, posture_score


# ── Training Functions ────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion_ce, criterion_mse, device):
    model.train()
    total_loss, total_ce, total_mse = 0, 0, 0
    correct = 0
    total = 0

    for batch in loader:
        if len(batch) == 3:
            x, y, posture = batch
            x, y, posture = x.to(device), y.to(device), posture.to(device)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            posture = None

        optimizer.zero_grad()
        logits, posture_pred = model(x)

        # Classification loss
        loss_ce = criterion_ce(logits, y)
        total_ce += loss_ce.item()

        # Posture loss (multi-task)
        if posture is not None:
            loss_mse = criterion_mse(posture_pred, posture)
            total_mse += loss_mse.item()
            loss = loss_ce + 0.5 * loss_mse  # Multi-task weighting
        else:
            loss = loss_ce

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = logits.max(1)
        correct += predicted.eq(y).sum().item()
        total += y.size(0)

    return {
        "loss": total_loss / len(loader),
        "ce_loss": total_ce / len(loader),
        "mse_loss": total_mse / len(loader) if posture is not None else 0,
        "acc": correct / total,
    }


@torch.no_grad()
def evaluate(model, loader, criterion_ce, criterion_mse, device):
    model.eval()
    total_loss, total_ce, total_mse = 0, 0, 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for batch in loader:
        if len(batch) == 3:
            x, y, posture = batch
            x, y, posture = x.to(device), y.to(device), posture.to(device)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            posture = None

        logits, posture_pred = model(x)
        loss_ce = criterion_ce(logits, y)
        total_ce += loss_ce.item()

        if posture is not None:
            loss_mse = criterion_mse(posture_pred, posture)
            total_mse += loss_mse.item()
            loss = loss_ce + 0.5 * loss_mse
        else:
            loss = loss_ce

        total_loss += loss.item()
        _, predicted = logits.max(1)
        correct += predicted.eq(y).sum().item()
        total += y.size(0)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    return {
        "loss": total_loss / len(loader),
        "ce_loss": total_ce / len(loader),
        "mse_loss": total_mse / len(loader) if posture is not None else 0,
        "acc": correct / total,
        "preds": all_preds,
        "labels": all_labels,
    }


def mc_dropout_inference(model, x, n_passes=30):
    """
    Monte Carlo Dropout for uncertainty estimation.
    Runs n_passes forward passes with dropout enabled.
    """
    model.train()  # Keep dropout active
    all_probs = []
    all_scores = []

    for _ in range(n_passes):
        with torch.no_grad():
            logits, posture = model(x)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_scores.append(posture.cpu().numpy())

    model.eval()

    probs_mean = np.mean(all_probs, axis=0)
    probs_std = np.std(all_probs, axis=0)
    scores_mean = np.mean(all_scores, axis=0)
    scores_std = np.std(all_scores, axis=0)

    return {
        "probs_mean": probs_mean,
        "probs_std": probs_mean,  # Shape: (batch, num_classes)
        "scores_mean": scores_mean,
        "scores_std": scores_std,
        "uncertainty": np.mean(probs_std, axis=1),  # Per-sample uncertainty
    }


# ── Main Training Script ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train BNN for posture analysis")
    parser.add_argument("--data", type=str, default="exercise_angles.csv",
                        help="Path to CSV with angle columns")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Using device: {args.device}")

    # ── Load Data ────────────────────────────────────────────────────────
    try:
        import pandas as pd
        df = pd.read_csv(args.data)
        print(f"Loaded {len(df)} samples from {args.data}")
    except FileNotFoundError:
        print(f"Dataset {args.data} not found. Generating synthetic data for demo...")
        # Generate synthetic data for demo purposes
        np.random.seed(42)
        n_samples = 5000
        n_classes = 5
        exercises = ["Jumping Jacks", "Pull ups", "Push Ups", "Russian twists", "Squats"]

        data = []
        for i in range(n_samples):
            exercise = exercises[i % n_classes]
            row = {
                "Shoulder_Angle": np.random.uniform(30, 170),
                "Elbow_Angle": np.random.uniform(50, 170),
                "Hip_Angle": np.random.uniform(60, 175),
                "Knee_Angle": np.random.uniform(80, 180),
                "Ankle_Angle": np.random.uniform(100, 180),
                "Shoulder_Ground_Angle": np.random.uniform(0, 90),
                "Elbow_Ground_Angle": np.random.uniform(0, 90),
                "Hip_Ground_Angle": np.random.uniform(0, 90),
                "Knee_Ground_Angle": np.random.uniform(0, 90),
                "Ankle_Ground_Angle": np.random.uniform(0, 90),
                "exercise": exercise,
            }
            data.append(row)
        df = pd.DataFrame(data)

    # ── Feature Engineering ─────────────────────────────────────────────
    feature_cols = [
        "Shoulder_Angle", "Elbow_Angle", "Hip_Angle", "Knee_Angle", "Ankle_Angle",
        "Shoulder_Ground_Angle", "Elbow_Ground_Angle", "Hip_Ground_Angle",
        "Knee_Ground_Angle", "Ankle_Ground_Angle",
    ]

    features = build_features(df[feature_cols])
    labels = df["exercise"].values

    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    # Create sequences
    X, y = create_sequences(features_scaled, labels_encoded)
    print(f"Created {len(X)} sequences of length {SEQ_LEN}")

    # Generate pseudo posture scores for multi-task learning
    posture_scores = []
    for i in range(len(df) - SEQ_LEN + 1):
        exercise = labels[i + SEQ_LEN - 1]
        angles = {col: df.iloc[i + SEQ_LEN - 1][col] for col in feature_cols}
        score = estimate_posture_score(angles, exercise)
        posture_scores.append(score)
    posture_scores = np.array(posture_scores)

    # Train/val split
    X_train, X_val, y_train, y_val, ps_train, ps_val = train_test_split(
        X, y, posture_scores, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # Datasets and loaders
    train_dataset = PostureDataset(X_train, y_train, ps_train)
    val_dataset = PostureDataset(X_val, y_val, ps_val)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # ── Model Setup ───────────────────────────────────────────────────────
    device = torch.device(args.device)
    model = PostureBNN().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()

    # ── Training Loop ────────────────────────────────────────────────────
    best_val_acc = 0
    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion_ce, criterion_mse, device)
        val_metrics = evaluate(model, val_loader, criterion_ce, criterion_mse, device)

        scheduler.step(val_metrics["loss"])

        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.4f}")

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_accuracy": best_val_acc,
                "classes": le.classes_.tolist(),
                "feature_names": [
                    "Shoulder_Angle", "Elbow_Angle", "Hip_Angle", "Knee_Angle", "Ankle_Angle",
                    "Shoulder_Ground_Angle", "Elbow_Ground_Angle", "Hip_Ground_Angle",
                    "Knee_Ground_Angle", "Ankle_Ground_Angle",
                    "shoulder_elbow_ratio", "hip_knee_ratio", "full_body_angle_sum",
                    "upper_body_diff", "lower_body_diff",
                ],
            }, "bnn_model.pt")
            print(f"  [Saved] Best model: {best_val_acc*100:.2f}%")

    # ── Save Artifacts ────────────────────────────────────────────────────
    joblib.dump(scaler, "bnn_scaler.pkl")
    joblib.dump(le, "bnn_label_encoder.pkl")

    # Save metadata
    confusion_matrix = [[0]*5 for _ in range(5)]
    for p, l in zip(val_metrics["preds"], val_metrics["labels"]):
        confusion_matrix[l][p] += 1

    metadata = {
        "model_type": "CNN-BiLSTM-BNN",
        "val_accuracy": best_val_acc,
        "confusion_matrix": confusion_matrix,
        "classes": le.classes_.tolist(),
        "sequence_length": SEQ_LEN,
        "mc_passes": 30,
    }
    with open("bnn_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*55)
    print("  TRAINING COMPLETE")
    print(f"  Best Val Accuracy: {best_val_acc*100:.2f}%")
    print(f"  Model saved: bnn_model.pt")
    print(f"  Scaler saved: bnn_scaler.pkl")
    print(f"  Encoder saved: bnn_label_encoder.pkl")
    print("="*55)


if __name__ == "__main__":
    main()
