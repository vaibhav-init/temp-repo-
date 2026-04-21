#!/usr/bin/env python3
"""
MLP Training Script for Crash Probability Prediction
=====================================================

Trains a Multi-Layer Perceptron (MLP) on kinematic features collected
from CARLA to predict the probability of collision within the next 2 seconds.

Input features (7):
  1. ego_speed           (m/s)
  2. ego_acceleration    (m/s²)
  3. nearest_distance    (m)
  4. relative_velocity   (m/s, positive = closing)
  5. ttc                 (seconds, capped at 10)
  6. obstacle_speed      (m/s)
  7. obstacle_type       (0=vehicle, 1=pedestrian, 2=none)

Output:
  crash_probability     (0.0 to 1.0)

Usage:
    python train_mlp.py
    python train_mlp.py --data dataset_crash/data.csv --epochs 200 --lr 0.001
    python train_mlp.py --data dataset_crash/data.csv --batch-size 128 --hidden 128 64 32
"""

import os
import argparse
import numpy as np
import pandas as pd
import pickle
import json
import math
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix, f1_score, accuracy_score
)


# ============================================================================
# Configuration
# ============================================================================
FEATURE_COLUMNS = [
    'ego_speed', 'ego_acceleration', 'nearest_distance',
    'relative_velocity', 'ttc', 'obstacle_speed', 'obstacle_type'
]
LABEL_COLUMN = 'collision_within_2s'

DEFAULT_DATA_PATH = 'dataset_crash/data.csv'
DEFAULT_MODEL_DIR = 'models'


# ============================================================================
# Dataset
# ============================================================================
class CrashDataset(Dataset):
    """PyTorch Dataset for crash probability prediction."""

    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ============================================================================
# MLP Model
# ============================================================================
class CrashMLP(nn.Module):
    """
    Multi-Layer Perceptron for crash probability prediction.

    Architecture:
        Input (7) → Hidden layers with ReLU + Dropout → Sigmoid output (1)

    The model outputs a single probability value [0, 1] representing
    the likelihood of collision within the next 2 seconds.
    """

    def __init__(self, input_dim=7, hidden_dims=None, dropout=0.3):
        super(CrashMLP, self).__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32, 16]

        layers = []
        prev_dim = input_dim

        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_dim))
            # Decrease dropout as we go deeper
            drop_rate = max(0.1, dropout - 0.1 * i)
            layers.append(nn.Dropout(drop_rate))
            prev_dim = h_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ============================================================================
# Training Functions
# ============================================================================
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


def evaluate(model, dataloader, criterion, device):
    """Evaluate model. Returns loss, predictions, and true labels."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            num_batches += 1

            all_preds.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    avg_loss = total_loss / max(1, num_batches)
    return avg_loss, np.array(all_preds), np.array(all_labels)


def find_optimal_threshold(y_true, y_pred):
    """Find threshold that maximizes F1 score."""
    best_f1 = 0
    best_thresh = 0.5

    for thresh in np.arange(0.1, 0.9, 0.01):
        preds = (y_pred >= thresh).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return best_thresh, best_f1


# ============================================================================
# Main Training Pipeline
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Train MLP for crash probability prediction')
    parser.add_argument('--data', default=DEFAULT_DATA_PATH, help='Path to CSV data file')
    parser.add_argument('--model-dir', default=DEFAULT_MODEL_DIR, help='Directory to save model')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden', nargs='+', type=int, default=[64, 32, 16],
                        help='Hidden layer dimensions')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--weight-ratio', type=float, default=0.0,
                        help='Positive class weight ratio (0=auto-compute from data)')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (epochs)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("CRASH PROBABILITY MLP TRAINER")
    print("=" * 70)
    print(f"  Data:         {args.data}")
    print(f"  Device:       {device}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Learning rate:{args.lr}")
    print(f"  Hidden dims:  {args.hidden}")
    print(f"  Dropout:      {args.dropout}")
    print(f"  Patience:     {args.patience}")
    print(f"  Seed:         {args.seed}")
    print("=" * 70)

    # ----------------------------------------------------------
    # 1. Load and inspect data
    # ----------------------------------------------------------
    print(f"\n📂 Loading data from {args.data}...")

    if not os.path.exists(args.data):
        print(f"  ❌ File not found: {args.data}")
        print(f"     Run data_collector_crash.py first to collect data.")
        return

    df = pd.read_csv(args.data)
    print(f"  ✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")

    # Check required columns
    missing = [c for c in FEATURE_COLUMNS + [LABEL_COLUMN] if c not in df.columns]
    if missing:
        print(f"  ❌ Missing columns: {missing}")
        return

    # ----------------------------------------------------------
    # 2. Data cleaning
    # ----------------------------------------------------------
    print(f"\n🧹 Cleaning data...")

    initial_count = len(df)

    # Replace infinities with NaN, then drop
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=FEATURE_COLUMNS + [LABEL_COLUMN], inplace=True)

    # Cap extreme values
    df['ttc'] = df['ttc'].clip(0, 10)
    df['ego_speed'] = df['ego_speed'].clip(0, 50)
    df['ego_acceleration'] = df['ego_acceleration'].clip(-20, 20)
    df['nearest_distance'] = df['nearest_distance'].clip(0, 50)
    df['relative_velocity'] = df['relative_velocity'].clip(-20, 20)
    df['obstacle_speed'] = df['obstacle_speed'].clip(0, 50)

    dropped = initial_count - len(df)
    print(f"  Dropped {dropped} rows with NaN/inf ({dropped/max(1,initial_count)*100:.1f}%)")
    print(f"  Remaining: {len(df):,} rows")

    # ----------------------------------------------------------
    # 3. Analyze class distribution
    # ----------------------------------------------------------
    print(f"\n📊 Class distribution:")
    class_counts = df[LABEL_COLUMN].value_counts()
    for label, count in class_counts.items():
        pct = count / len(df) * 100
        print(f"  Class {int(label)}: {count:,} ({pct:.1f}%)")

    n_pos = int(class_counts.get(1, 0))
    n_neg = int(class_counts.get(0, 0))

    if n_pos == 0:
        print(f"\n  ⚠️  WARNING: No positive samples found!")
        print(f"     The model cannot learn without collision data.")
        print(f"     Run more scenarios or adjust TM settings for more crashes.")
        return

    # ----------------------------------------------------------
    # 4. Feature statistics
    # ----------------------------------------------------------
    print(f"\n📊 Feature statistics:")
    for col in FEATURE_COLUMNS:
        vals = df[col]
        print(f"  {col:22s}  min={vals.min():8.2f}  max={vals.max():8.2f}  "
              f"mean={vals.mean():8.2f}  std={vals.std():8.2f}")

    # ----------------------------------------------------------
    # 5. Prepare features and labels
    # ----------------------------------------------------------
    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    y = df[LABEL_COLUMN].values.astype(np.float32)

    print(f"\n  Feature matrix: {X.shape}")
    print(f"  Label vector:   {y.shape}")

    # ----------------------------------------------------------
    # 6. Train / Validation / Test split (70/15/15, stratified)
    # ----------------------------------------------------------
    print(f"\n✂️  Splitting data (70/15/15, stratified)...")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=args.seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=args.seed, stratify=y_temp)

    print(f"  Train: {len(X_train):,} (pos: {int(y_train.sum()):,})")
    print(f"  Val:   {len(X_val):,}   (pos: {int(y_val.sum()):,})")
    print(f"  Test:  {len(X_test):,}  (pos: {int(y_test.sum()):,})")

    # ----------------------------------------------------------
    # 7. Feature normalization (StandardScaler)
    # ----------------------------------------------------------
    print(f"\n📐 Fitting StandardScaler on training data...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f"  Scale means:  {scaler.mean_}")
    print(f"  Scale stds:   {scaler.scale_}")

    # ----------------------------------------------------------
    # 8. Create DataLoaders
    # ----------------------------------------------------------
    train_dataset = CrashDataset(X_train_scaled, y_train)
    val_dataset = CrashDataset(X_val_scaled, y_val)
    test_dataset = CrashDataset(X_test_scaled, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")

    # ----------------------------------------------------------
    # 9. Build model
    # ----------------------------------------------------------
    model = CrashMLP(
        input_dim=len(FEATURE_COLUMNS),
        hidden_dims=args.hidden,
        dropout=args.dropout
    ).to(device)

    print(f"\n🧠 Model architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # ----------------------------------------------------------
    # 10. Loss function with class weighting
    # ----------------------------------------------------------
    if args.weight_ratio > 0:
        pos_weight_val = args.weight_ratio
    else:
        # Auto-compute: weight = n_negative / n_positive
        pos_weight_val = n_neg / max(1, n_pos)
        pos_weight_val = min(pos_weight_val, 50.0)  # Cap at 50x

    pos_weight = torch.FloatTensor([pos_weight_val]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Since our model already has Sigmoid, we use BCELoss but with manual weighting
    # Actually, let's use BCELoss with manual sample weighting instead
    # to work correctly with our Sigmoid output
    criterion = nn.BCELoss(reduction='none')

    print(f"\n⚖️  Class weights:")
    print(f"  Positive weight: {pos_weight_val:.2f}x (to handle class imbalance)")
    print(f"  Negative:Positive ratio = {n_neg}:{n_pos} = {n_neg/max(1,n_pos):.1f}:1")

    # ----------------------------------------------------------
    # 11. Optimizer and scheduler
    # ----------------------------------------------------------
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    print(f"\n🔧 Optimizer: Adam (lr={args.lr}, weight_decay=1e-4)")
    print(f"   Scheduler: ReduceLROnPlateau (factor=0.5, patience=10)")

    # ----------------------------------------------------------
    # 12. Training loop
    # ----------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"TRAINING ({args.epochs} epochs)")
    print(f"{'=' * 70}")

    best_val_loss = float('inf')
    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_aucs = []

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        n_batches = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)

            # Weighted loss
            raw_loss = criterion(outputs, labels)
            weights = torch.where(labels == 1, pos_weight_val, 1.0)
            loss = (raw_loss * weights).mean()

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        avg_train_loss = train_loss / max(1, n_batches)

        # Validate
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        val_preds = []
        val_labels_list = []

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)

                outputs = model(features)
                raw_loss = criterion(outputs, labels)
                weights = torch.where(labels == 1, pos_weight_val, 1.0)
                loss = (raw_loss * weights).mean()

                val_loss += loss.item()
                n_val_batches += 1

                val_preds.extend(outputs.cpu().numpy().flatten())
                val_labels_list.extend(labels.cpu().numpy().flatten())

        avg_val_loss = val_loss / max(1, n_val_batches)
        val_preds_arr = np.array(val_preds)
        val_labels_arr = np.array(val_labels_list)

        # Compute AUC-ROC (need both classes present)
        try:
            val_auc = roc_auc_score(val_labels_arr, val_preds_arr)
        except ValueError:
            val_auc = 0.0

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_aucs.append(val_auc)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Print progress
        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch [{epoch:4d}/{args.epochs}]  "
                  f"Train Loss: {avg_train_loss:.4f}  "
                  f"Val Loss: {avg_val_loss:.4f}  "
                  f"Val AUC: {val_auc:.4f}  "
                  f"LR: {current_lr:.6f}")

        # Early stopping / best model tracking
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            os.makedirs(args.model_dir, exist_ok=True)
            model_path = os.path.join(args.model_dir, 'crash_mlp.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_dim': len(FEATURE_COLUMNS),
                'hidden_dims': args.hidden,
                'dropout': args.dropout,
                'feature_columns': FEATURE_COLUMNS,
                'epoch': epoch,
                'val_loss': avg_val_loss,
                'val_auc': val_auc,
            }, model_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  ⏹️  Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    print(f"\n  ✅ Training complete!")
    print(f"     Best epoch: {best_epoch}")
    print(f"     Best val loss: {best_val_loss:.4f}")
    print(f"     Best val AUC: {best_val_auc:.4f}")

    # ----------------------------------------------------------
    # 13. Save scaler
    # ----------------------------------------------------------
    scaler_path = os.path.join(args.model_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  💾 Scaler saved to {scaler_path}")

    model_path = os.path.join(args.model_dir, 'crash_mlp.pth')
    print(f"  💾 Model saved to {model_path}")

    # ----------------------------------------------------------
    # 14. Load best model and evaluate on test set
    # ----------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"TEST SET EVALUATION")
    print(f"{'=' * 70}")

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    test_preds = []
    test_labels_list = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features)
            test_preds.extend(outputs.cpu().numpy().flatten())
            test_labels_list.extend(labels.cpu().numpy().flatten())

    test_preds_arr = np.array(test_preds)
    test_labels_arr = np.array(test_labels_list)

    # AUC-ROC
    try:
        test_auc = roc_auc_score(test_labels_arr, test_preds_arr)
    except ValueError:
        test_auc = 0.0

    # Average Precision (PR-AUC)
    try:
        test_ap = average_precision_score(test_labels_arr, test_preds_arr)
    except ValueError:
        test_ap = 0.0

    # Find optimal threshold
    optimal_thresh, optimal_f1 = find_optimal_threshold(test_labels_arr, test_preds_arr)

    print(f"\n  📊 Test Metrics:")
    print(f"     AUC-ROC:                {test_auc:.4f}")
    print(f"     Average Precision (AP): {test_ap:.4f}")
    print(f"     Optimal threshold:      {optimal_thresh:.2f}")
    print(f"     F1 @ optimal thresh:    {optimal_f1:.4f}")

    # Classification report at optimal threshold
    test_preds_binary = (test_preds_arr >= optimal_thresh).astype(int)
    test_acc = accuracy_score(test_labels_arr, test_preds_binary)

    print(f"     Accuracy @ {optimal_thresh:.2f}:       {test_acc:.4f}")

    print(f"\n  📊 Classification Report (threshold={optimal_thresh:.2f}):")
    print(classification_report(
        test_labels_arr, test_preds_binary,
        target_names=['Safe (0)', 'Crash (1)'],
        digits=4
    ))

    # Confusion matrix
    cm = confusion_matrix(test_labels_arr, test_preds_binary)
    print(f"  📊 Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                  Safe    Crash")
    print(f"     Actual Safe  {cm[0][0]:6d}  {cm[0][1]:6d}")
    print(f"     Actual Crash {cm[1][0]:6d}  {cm[1][1]:6d}")

    tn, fp, fn, tp = cm.ravel()
    print(f"\n     True Positives:  {tp:6d}  (correctly predicted crashes)")
    print(f"     True Negatives:  {tn:6d}  (correctly predicted safe)")
    print(f"     False Positives: {fp:6d}  (false alarms)")
    print(f"     False Negatives: {fn:6d}  (missed crashes)")

    if tp + fn > 0:
        recall = tp / (tp + fn)
        print(f"     Recall (sensitivity): {recall:.4f}  ({recall*100:.1f}% of crashes detected)")

    if tp + fp > 0:
        precision = tp / (tp + fp)
        print(f"     Precision:            {precision:.4f}  ({precision*100:.1f}% of alarms are real)")

    # ----------------------------------------------------------
    # 15. Prediction distribution analysis
    # ----------------------------------------------------------
    print(f"\n  📊 Prediction Distribution:")
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        above = (test_preds_arr >= threshold).sum()
        print(f"     P >= {threshold:.1f}: {above:6d} samples ({above/len(test_preds_arr)*100:.1f}%)")

    # ----------------------------------------------------------
    # 16. Save training metadata
    # ----------------------------------------------------------
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'data_path': args.data,
        'total_samples': len(df),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'feature_columns': FEATURE_COLUMNS,
        'hidden_dims': args.hidden,
        'dropout': args.dropout,
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'best_epoch': best_epoch,
        'total_epochs_trained': epoch,
        'test_auc_roc': float(test_auc),
        'test_avg_precision': float(test_ap),
        'optimal_threshold': float(optimal_thresh),
        'optimal_f1': float(optimal_f1),
        'test_accuracy': float(test_acc),
        'pos_weight': float(pos_weight_val),
        'class_distribution': {
            'negative': int(n_neg),
            'positive': int(n_pos),
            'ratio': float(n_neg / max(1, n_pos)),
        },
        'scaler_means': scaler.mean_.tolist(),
        'scaler_stds': scaler.scale_.tolist(),
    }

    metadata_path = os.path.join(args.model_dir, 'training_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  💾 Training metadata saved to {metadata_path}")

    # ----------------------------------------------------------
    # 17. Summary
    # ----------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"TRAINING COMPLETE — SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Model:          {model_path}")
    print(f"  Scaler:         {scaler_path}")
    print(f"  Metadata:       {metadata_path}")
    print(f"  Features:       {FEATURE_COLUMNS}")
    print(f"  Architecture:   {len(FEATURE_COLUMNS)} → {' → '.join(map(str, args.hidden))} → 1")
    print(f"  Test AUC-ROC:   {test_auc:.4f}")
    print(f"  Test AP:        {test_ap:.4f}")
    print(f"  Best threshold: {optimal_thresh:.2f}")
    print(f"  Best F1:        {optimal_f1:.4f}")
    print(f"{'=' * 70}")
    print(f"\n  To run live inference:")
    print(f"    python crash_predictor_live.py --model {model_path} --scaler {scaler_path}")
    print()


if __name__ == '__main__':
    main()
