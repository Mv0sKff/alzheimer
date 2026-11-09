import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from args import get_args
from dataset import SubjectSampler, alzheimers_dataset, first_scan
from model import MyModel
from summary import fold_number, summarize_folds
from train import train_model
from utils import seed_everything

# Show each log line immediately, also when stdout is redirected to a file.
sys.stdout.reconfigure(line_buffering=True)  # type: ignore

# 1. get arguments
args = get_args()
print("Epochs:", args.epochs)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
run_name = datetime.now().astimezone().strftime("%Y-%m-%d_%H%M%S")
run_dir = Path(args.resume) if args.resume else Path(args.output_dir) / run_name

# only the worker count may be adjusted between resumes
run_dir.mkdir(parents=True, exist_ok=True)
config = {k: v for k, v in vars(args).items() if k not in ("resume", "output_dir")}
config_path = run_dir / "args.json"
if config_path.exists():
    saved = json.loads(config_path.read_text())
    changed = sorted(
        k
        for k in config.keys() | saved.keys()
        if k != "num_workers" and config.get(k) != saved.get(k)
    )
    if changed:
        raise SystemExit(f"Arguments differ from {config_path}: {changed}")
config_path.write_text(json.dumps(config, indent=2) + "\n")

loader_options = {
    "batch_size": args.batch_size,
    "num_workers": args.num_workers,
    "pin_memory": device.type == "cuda",
    "persistent_workers": args.num_workers > 0,
}

# 2. Iterate over each fold

folds = sorted(fold_number(p) for p in Path(args.csv_dir).glob("train_fold_*.csv"))
if not folds:
    raise SystemExit(f"No train_fold_*.csv files in {args.csv_dir}")

for fold in folds:
    print(f"fold {fold}")
    seed_everything(args.seed + fold)
    generator = torch.Generator().manual_seed(args.seed + fold)

    train_set = pd.read_csv(os.path.join(args.csv_dir, f"train_fold_{fold}.csv"))
    # The repeated scans of a subject are nearly identical, so one is enough.
    val_set = first_scan(
        pd.read_csv(os.path.join(args.csv_dir, f"val_fold_{fold}.csv"))
    )

    # 3. Preparing dataset
    train_dataset = alzheimers_dataset(train_set, augment=args.augment)
    val_dataset = alzheimers_dataset(val_set)

    # 4. Create data loaders
    train_loader = DataLoader(
        train_dataset,
        sampler=SubjectSampler(
            train_dataset.subjects, args.slices_per_subject, generator
        ),
        drop_last=True,
        generator=generator,
        **loader_options,
    )
    val_loader = DataLoader(val_dataset, **loader_options)

    # 5. Initializing the model
    model = MyModel(backbone=args.backbone).to(device)

    # 6. Train the model
    train_model(
        model,
        train_loader,
        val_loader,
        fold=fold,
        output_dir=run_dir,
        args=args,
        save=True,
    )

# 7. Summarize the validation results across folds
summarize_folds(run_dir)
