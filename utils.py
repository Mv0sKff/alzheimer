from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)


def compute_metrics(labels, logits):
    """Return accuracy, balanced accuracy, macro ROC-AUC and macro AP (one-vs-rest)."""
    probs = torch.softmax(logits.double(), dim=1).numpy()
    labels = labels.numpy()
    predictions = probs.argmax(axis=1)
    one_hot = np.eye(probs.shape[1])[labels]

    accuracy = float((predictions == labels).mean())
    balanced_acc = balanced_accuracy_score(labels, predictions)
    roc_auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    avg_precision = average_precision_score(one_hot, probs, average="macro")
    return accuracy, balanced_acc, roc_auc, avg_precision


def plot_metric(train_values, val_values, name, save_path=None):
    if len(train_values) != len(val_values):
        raise ValueError("Training and validation histories must have the same length")

    figure = Figure()
    axis = figure.add_subplot()
    epochs = range(1, len(train_values) + 1)
    axis.plot(epochs, train_values, label=f"Training {name}")
    axis.plot(epochs, val_values, label=f"Validation {name}")
    axis.set_xlabel("Epoch")
    axis.set_ylabel(name)
    axis.legend()
    figure.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path)
    return figure
