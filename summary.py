import sys
from pathlib import Path

import pandas as pd
import torch

from schemas import METRICS, History, best_epoch


def fold_number(path):
    return int(path.stem.rsplit("_", 1)[1])


def summary_table(rows):
    """Per-fold rows followed by the mean and sample standard deviation of each metric."""
    table = pd.DataFrame.from_dict(rows, orient="index")
    metrics = table.drop(columns="best_epoch", errors="ignore")
    summary = pd.concat([table, metrics.agg(["mean", "std"])])
    if "best_epoch" in summary:
        summary["best_epoch"] = summary["best_epoch"].astype("Int64")
    return summary


def summarize_folds(run_dir):
    """Write and print the validation metrics of each fold's best epoch."""
    run_dir = Path(run_dir)
    rows = {}
    for path in sorted(run_dir.glob("last_fold_*.pt"), key=fold_number):
        state = torch.load(path, map_location="cpu")
        history = History(**state["val_history"])
        epoch = best_epoch(history)
        rows[f"fold_{fold_number(path)}"] = {
            "best_epoch": epoch,
            **{name: getattr(history, name)[epoch] for name in ["loss", *METRICS]},
        }
    if not rows:
        raise SystemExit(f"No last_fold_*.pt checkpoints in {run_dir}")

    summary = summary_table(rows)
    summary.to_csv(run_dir / "val_summary.csv", float_format="%.4f")
    print("Validation metrics at each fold's best epoch:")
    print(summary.T.to_string(float_format="{:.4f}".format))
    return summary


if __name__ == "__main__":
    summarize_folds(sys.argv[1])
