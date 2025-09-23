"""
gnn_model_enhanced.py

Enhanced GNN training/eval pipeline for HR attrition node classification.

Features:
- Supports GCN, GAT, GraphSAGE (SAGEConv) architectures
- Class-weighted CrossEntropyLoss to penalize minority class
- Configurable number of layers, hidden dims, dropout, weight decay, lr, epochs
- Threshold tuning + CSV + plot of Precision / Recall / F1 vs threshold
- Saves classification reports and confusion matrices to reports/
- Prints graph summary (nodes, edges, avg degree) via EmployeeGraphBuilder.to_pyg()
"""

import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

# PyG imports
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

# Add src to path for local imports
script_path = Path(__file__).resolve()
src_path = script_path.parent.parent
sys.path.append(str(src_path))

# Local modules (make sure paths are correct in your repo)
from analysis.data_analysis_sim import AdvancedEDA, FILE_PATH
from analysis.gnn_preprocessing import EmployeeGraphBuilder


# -----------------------------
# Model definitions
# -----------------------------
class BaseGNN(nn.Module):
    """Wrapper to create multi-layer GNN using specified conv layer type."""

    def __init__(
        self,
        conv_type,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        dropout=0.5,
        heads=1,
    ):
        super().__init__()
        assert num_layers >= 2, "num_layers must be >= 2"

        self.convs = nn.ModuleList()
        self.dropout = dropout
        self.num_layers = num_layers
        self.conv_type = conv_type.lower()
        self.heads = heads

        # first layer
        self.convs.append(self._make_conv(conv_type, in_channels, hidden_channels, heads=heads))

        # middle layers
        for _ in range(num_layers - 2):
            self.convs.append(self._make_conv(conv_type, hidden_channels, hidden_channels, heads=heads))

        # final layer
        self.convs.append(self._make_conv(conv_type, hidden_channels, out_channels, heads=1))

    def _make_conv(self, conv_type, in_ch, out_ch, heads=1):
        if conv_type == "gcn":
            return GCNConv(in_ch, out_ch)
        elif conv_type == "gat":
            # concat=False ensures output dim = out_ch, not out_ch*heads
            return GATConv(
                in_channels=in_ch,
                out_channels=out_ch,
                heads=heads,
                concat=False,
                dropout=0.6,
            )
        elif conv_type == "sage":
            return SAGEConv(in_ch, out_ch)
        else:
            raise ValueError("Unsupported conv_type. Choose 'gcn', 'gat', or 'sage'.")


    def forward(self, x, edge_index):
        # Apply sequential conv layers
        for i, conv in enumerate(self.convs):
            if self.conv_type == "gat" and isinstance(conv, GATConv):
                x = conv(x, edge_index)
            else:
                x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# -----------------------------
# Training & evaluation helpers
# -----------------------------
def compute_class_weights(y):
    """
    Compute weights for CrossEntropyLoss: inverse frequency per class.
    y: numpy array of 0/1 labels
    Returns tensor of shape (2,)
    """
    classes, counts = np.unique(y, return_counts=True)
    # ensure order [0,1]
    freq = {c: cnt for c, cnt in zip(classes, counts)}
    n0 = freq.get(0, 0)
    n1 = freq.get(1, 0)
    if n0 == 0 or n1 == 0:
        # fallback
        return torch.tensor([1.0, 1.0], dtype=torch.float)
    w0 = (n0 + n1) / (2.0 * n0)
    w1 = (n0 + n1) / (2.0 * n1)
    return torch.tensor([w0, w1], dtype=torch.float)


def train_epoch(model, data, optimizer, criterion, train_mask):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate_probs(model, data, mask):
    """
    Return predicted probabilities and labels on nodes specified by mask.
    """
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)  # shape [N, out_channels]
        probs = F.softmax(logits[mask], dim=1)[:, 1].cpu().numpy()
        preds = logits[mask].argmax(dim=1).cpu().numpy()
        labels = data.y[mask].cpu().numpy()
    return probs, preds, labels


def save_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar()
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    ticks = np.arange(2)
    plt.xticks(ticks, ["No Attrition", "Attrition"])
    plt.yticks(ticks, ["No Attrition", "Attrition"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def threshold_tuning_and_save(probs, labels, model_name="gnn", report_dir="reports", thresholds=None):
    """
    Evaluate precision/recall/F1 across thresholds, save CSV + plot, return best thresholds.
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)

    results = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        results.append({"Threshold": float(t), "Precision": float(precision), "Recall": float(recall), "F1": float(f1)})

    results_df = pd.DataFrame(results)
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(report_dir) / f"{model_name}_threshold_eval.csv"
    results_df.to_csv(csv_path, index=False)

    # Plot curves
    plt.figure(figsize=(8, 6))
    plt.plot(results_df["Threshold"], results_df["Precision"], marker="o", label="Precision")
    plt.plot(results_df["Threshold"], results_df["Recall"], marker="s", label="Recall")
    plt.plot(results_df["Threshold"], results_df["F1"], marker="^", label="F1")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"{model_name.upper()} Threshold Tuning: Precision / Recall / F1")
    plt.legend()
    plot_path = Path(report_dir) / f"{model_name}_threshold_curves.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    # bests
    best_f1 = results_df.loc[results_df["F1"].idxmax()].to_dict()
    best_recall = results_df.loc[results_df["Recall"].idxmax()].to_dict()

    return results_df, best_f1, best_recall, csv_path, plot_path


# -----------------------------
# Main training pipeline
# -----------------------------
def main(
    model_type="gcn",  # "gcn", "gat", "sage"
    num_layers=3,
    hidden_channels=64,
    heads=2,  # used for GAT
    dropout=0.5,
    lr=0.005,
    weight_decay=5e-4,
    epochs=100,
    device="cpu",
    k_neighbors=5,
    report_dir="reports",
    seed=42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("Loading dataset and building GNN graph...")
    eda = AdvancedEDA(data_path=FILE_PATH)
    graph_builder = EmployeeGraphBuilder(eda.df)
    data = graph_builder.to_pyg(k=k_neighbors)  # prints graph summary

    # Move data to device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    # Train/test node masks (stratified)
    idx = np.arange(data.num_nodes)
    train_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=data.y.cpu().numpy(), random_state=seed)
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    train_mask = train_mask.to(device)
    test_mask = test_mask.to(device)

    # Build model
    print(f"\nBuilding {model_type.upper()} model: layers={num_layers}, hidden={hidden_channels}, dropout={dropout}, heads={heads}")
    model = BaseGNN(conv_type=model_type, in_channels=data.num_node_features, hidden_channels=hidden_channels, out_channels=2, num_layers=num_layers, dropout=dropout, heads=heads)
    model = model.to(device)

    # Class weights (penalize minority class)
    labels_all = data.y.cpu().numpy()
    class_weights = compute_class_weights(labels_all).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optional scheduler: reduce on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)

    # Training loop
    best_val_auc = 0.0
    print("\nTraining...")
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, data, optimizer, criterion, train_mask)
        if epoch % 10 == 0 or epoch == 1:
            # quick eval on test
            probs_test, preds_test, labels_test = evaluate_probs(model, data, test_mask)
            try:
                auc = roc_auc_score(labels_test, probs_test)
            except ValueError:
                auc = 0.0
            pr_auc = average_precision_score(labels_test, probs_test) if labels_test.sum() > 0 else 0.0
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Test ROC AUC: {auc:.3f} | PR AUC: {pr_auc:.3f}")
            # scheduler step (monitor ROC AUC)
            scheduler.step(auc)
            print(f"[Scheduler] LR after step: {optimizer.param_groups[0]['lr']:.6f}")

    # Final evaluation: get probabilities on test mask
    probs_test, preds_test_default, labels_test = evaluate_probs(model, data, test_mask)
    try:
        final_auc = roc_auc_score(labels_test, probs_test)
    except ValueError:
        final_auc = 0.0
    final_pr_auc = average_precision_score(labels_test, probs_test) if labels_test.sum() > 0 else 0.0

    print("\nFinal evaluation on test nodes:")
    print("ROC AUC:", final_auc)
    print("PR AUC:", final_pr_auc)

    # Save default classification report (threshold 0.5)
    preds_default = (probs_test >= 0.5).astype(int)
    report_default = classification_report(labels_test, preds_default, target_names=["No Attrition", "Attrition"])
    print("\nClassification Report (threshold=0.5):\n", report_default)

    Path(report_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(report_dir) / f"gnn_{model_type}_report_default.txt", "w") as f:
        f.write(f"ROC AUC: {final_auc}\nPR AUC: {final_pr_auc}\n\n")
        f.write("Classification Report (threshold=0.5):\n")
        f.write(report_default)

    # Save confusion matrix for default threshold
    save_confusion_matrix(labels_test, preds_default, Path(report_dir) / f"gnn_{model_type}_confusion_default.png")

    # Threshold tuning + save CSV + plot
    results_df, best_f1, best_recall, csv_path, plot_path = threshold_tuning_and_save(probs_test, labels_test, model_name=f"gnn_{model_type}", report_dir=report_dir)
    print(f"Threshold tuning CSV: {csv_path}, plot: {plot_path}")
    print("Best threshold by F1:", best_f1)
    print("Best threshold by Recall:", best_recall)

    # Save classification report at best F1
    preds_best_f1 = (probs_test >= best_f1["Threshold"]).astype(int)
    report_bestf1 = classification_report(labels_test, preds_best_f1, target_names=["No Attrition", "Attrition"])
    with open(Path(report_dir) / f"gnn_{model_type}_report_bestF1.txt", "w") as f:
        f.write(f"Best threshold by F1: {best_f1}\n\n")
        f.write(report_bestf1)
    save_confusion_matrix(labels_test, preds_best_f1, Path(report_dir) / f"gnn_{model_type}_confusion_bestF1.png")

    print("\nDone. Reports and plots saved under:", report_dir)


if __name__ == "__main__":
    # Example call: change parameters here or call from CLI wrappers
    main(
        model_type="gat",     # "gcn", "gat", or "sage"
        num_layers=3,
        hidden_channels=64,
        heads=4,              # only used when model_type="gat"
        dropout=0.5,
        lr=0.005,
        weight_decay=5e-4,
        epochs=100,
        device="cuda" if torch.cuda.is_available() else "cpu",
        k_neighbors=5,
        report_dir="reports",
        seed=42,
    )
