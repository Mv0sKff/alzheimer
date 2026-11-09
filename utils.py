import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_rng_state(generator=None):
    """Return the RNG states that affect training, as tensors for a safe torch.load."""
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader": generator.get_state() if generator is not None else None,
    }


def set_rng_state(state, generator=None):
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if generator is not None and state["loader"] is not None:
        generator.set_state(state["loader"])


def classification_metrics(labels, probs):
    """Return accuracy, balanced accuracy, macro ROC-AUC and macro AP (one-vs-rest)."""
    predictions = probs.argmax(axis=1)
    one_hot = np.eye(probs.shape[1])[labels]

    accuracy = float((predictions == labels).mean())
    balanced_acc = float(balanced_accuracy_score(labels, predictions))
    roc_auc = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
    avg_precision = float(average_precision_score(one_hot, probs, average="macro"))
    return accuracy, balanced_acc, roc_auc, avg_precision


def subject_probabilities(labels, probs, subjects):
    """Average the slice probabilities of each subject.

    Returns the subject ids, their labels and their averaged probabilities.
    """
    subjects = np.asarray(subjects)
    per_subject = pd.DataFrame(probs).groupby(subjects).mean()
    subject_labels = pd.Series(labels).groupby(subjects).first()[per_subject.index]
    return (
        per_subject.index.to_numpy(),
        subject_labels.to_numpy(),
        per_subject.to_numpy(),
    )


def metrics_from_probs(labels, probs, subjects):
    """Return the slice-level metrics followed by the subject-level metrics."""
    _, subject_labels, subject_probs = subject_probabilities(labels, probs, subjects)
    return (
        *classification_metrics(labels, probs),
        *classification_metrics(subject_labels, subject_probs),
    )


def compute_metrics(labels, logits, subjects):
    probs = torch.softmax(logits.double(), dim=1).numpy()
    return metrics_from_probs(labels.numpy(), probs, subjects)


def write_history(path, lrs, train_history, val_history):
    """Write one row per epoch with the learning rate and all train and val metrics."""
    table = pd.DataFrame({"epoch": range(len(lrs)), "lr": lrs})
    for prefix, history in (("train", train_history), ("val", val_history)):
        for name, values in asdict(history).items():
            table[f"{prefix}_{name}"] = values
    path = Path(path)
    tmp_path = path.with_suffix(".tmp")
    table.to_csv(tmp_path, index=False, float_format="%.6g")
    tmp_path.replace(path)


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
