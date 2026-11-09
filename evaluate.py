import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from dataset import alzheimers_dataset
from model import MyModel
from schemas import CLASSES, METRICS
from summary import fold_number, summary_table
from utils import metrics_from_probs, subject_probabilities

ROOT = Path(__file__).resolve().parent


def predict(model, loader, use_amp=False):
    """Return the softmax probabilities of every slice in loader order."""
    model.eval()
    device = next(model.parameters()).device
    logits = []
    with torch.no_grad(), torch.autocast(device.type, enabled=use_amp):
        for batch in loader:
            logits.append(model(batch["image"].to(device, non_blocking=True)).float())
    return torch.softmax(torch.cat(logits).double(), dim=1).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--csv", type=Path, default=ROOT / "data/csv/test.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()

    config = json.loads((args.run_dir / "args.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config["amp"] and device.type == "cuda"
    model_paths = sorted(args.run_dir.glob("best_model_fold_*.pt"), key=fold_number)
    if not model_paths:
        raise SystemExit(f"No best_model_fold_*.pt files in {args.run_dir}")

    test_set = pd.read_csv(args.csv)
    labels = test_set["Label"].to_numpy()
    subjects = test_set["Subject"].to_numpy()
    loader = DataLoader(
        alzheimers_dataset(test_set),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    rows = {}
    fold_probs = []
    for path in model_paths:
        model = MyModel(backbone=config["backbone"], pretrained=False).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        probs = predict(model, loader, use_amp)
        fold_probs.append(probs)
        rows[f"fold_{fold_number(path)}"] = dict(
            zip(METRICS, metrics_from_probs(labels, probs, subjects))
        )
        print(f"evaluated {path.name}")

    # The ensemble averages the slice probabilities of all fold models.
    ensemble_probs = np.mean(fold_probs, axis=0)
    ensemble = dict(zip(METRICS, metrics_from_probs(labels, ensemble_probs, subjects)))
    results = pd.concat(
        [summary_table(rows), pd.DataFrame.from_dict({"ensemble": ensemble}, "index")]
    )
    results.to_csv(args.run_dir / "test_results.csv", float_format="%.4f")

    subject_ids, subject_labels, subject_probs = subject_probabilities(
        labels, ensemble_probs, subjects
    )
    subject_predictions = subject_probs.argmax(axis=1)
    predictions = pd.DataFrame(subject_probs, columns=[f"P({c})" for c in CLASSES])
    predictions.insert(0, "Subject", subject_ids)
    predictions.insert(1, "Class", [CLASSES[i] for i in subject_labels])
    predictions.insert(2, "Predicted", [CLASSES[i] for i in subject_predictions])
    predictions.to_csv(
        args.run_dir / "test_predictions.csv", index=False, float_format="%.4f"
    )

    matrix = confusion_matrix(
        subject_labels, subject_predictions, labels=range(len(CLASSES))
    )
    print(f"Test metrics ({args.csv.name}):")
    print(results.T.to_string(float_format="{:.4f}".format))
    print("Ensemble subject confusion matrix (rows: true, columns: predicted):")
    print(pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).to_string())


if __name__ == "__main__":
    main()
