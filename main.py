import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from args import *
from dataset import knee_xray_dataset
from model import MyModel
from train import train_model

# 1. get arguments
args = get_args()
print("Epochs:", args.epochs)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
run_name = datetime.now().astimezone().strftime("%Y-%m-%d_%H%M%S")
run_dir = Path(args.resume) if args.resume else Path(args.output_dir) / run_name

# 2. Iterate over each fold

for fold in range(5):
    print(f"fold {fold}")
    train_set = pd.read_csv(os.path.join(args.csv_dir, f"train_fold_{fold}.csv"))
    val_set = pd.read_csv(os.path.join(args.csv_dir, f"val_fold_{fold}.csv"))

    # 3. Preparing dataset
    train_dataset = knee_xray_dataset(train_set)
    val_dataset = knee_xray_dataset(val_set)

    # 4. Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # 5. Initializing the model
    model = MyModel(backbone=args.backbone).to(device)

    # 6. Train the model
    train_model(
        model,
        train_loader,
        val_loader,
        fold=fold,
        output_dir=run_dir,
        save=True,
    )
