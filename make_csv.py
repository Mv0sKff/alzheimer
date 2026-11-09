import argparse
import os
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from schemas import CLASSES

ROOT = Path(__file__).resolve().parent
FILENAME_RE = re.compile(r"^(OAS1_\d{4})_MR\d+_(mpr-\d+)_(\d+)\.jpg$")


def scan_images(data_dir):
    rows = []
    for label, cls in enumerate(CLASSES):
        for path in sorted((data_dir / cls).glob("*.jpg")):
            match = FILENAME_RE.match(path.name)
            if match is None:
                raise ValueError(f"Unexpected filename: {path}")
            subject, scan, slice_idx = match.groups()
            rows.append(
                {
                    "Path": Path(os.path.relpath(path, ROOT)).as_posix(),
                    "Subject": subject,
                    "Scan": scan,
                    "Slice": int(slice_idx),
                    "Label": label,
                    "Class": cls,
                }
            )
    return pd.DataFrame(rows)


def summarize(name, df):
    counts = df.groupby("Class").agg(
        subjects=("Subject", "nunique"), images=("Path", "size")
    )
    counts = counts.reindex(CLASSES, fill_value=0)
    summary = ", ".join(
        f"{cls}: {row.subjects} subj / {row.images} img"
        for cls, row in counts.iterrows()
    )
    print(f"{name:<14} {summary}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out_dir", type=Path, default=ROOT / "data/csv")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    images = scan_images(args.data_dir)
    subjects = images.groupby("Subject")["Label"].unique()
    conflicting = subjects[subjects.map(len) > 1]
    if not conflicting.empty:
        raise ValueError(f"Subjects in multiple classes: {list(conflicting.index)}")
    subjects = subjects.map(lambda labels: labels[0]).rename("Label").reset_index()

    dev_subjects, test_subjects = train_test_split(
        subjects,
        test_size=args.test_size,
        stratify=subjects["Label"],
        random_state=args.seed,
    )
    dev_subjects = dev_subjects.reset_index(drop=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    test = images[images["Subject"].isin(test_subjects["Subject"])]
    test.to_csv(args.out_dir / "test.csv", index=False)
    summarize("test", test)

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(dev_subjects, dev_subjects["Label"])
    ):
        train_ids = dev_subjects["Subject"].iloc[train_idx]
        val_ids = dev_subjects["Subject"].iloc[val_idx]
        train = images[images["Subject"].isin(train_ids)]
        val = images[images["Subject"].isin(val_ids)]
        train.to_csv(args.out_dir / f"train_fold_{fold}.csv", index=False)
        val.to_csv(args.out_dir / f"val_fold_{fold}.csv", index=False)
        summarize(f"train_fold_{fold}", train)
        summarize(f"val_fold_{fold}", val)


if __name__ == "__main__":
    main()
