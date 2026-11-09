import argparse
import os
from pathlib import Path

from model import BACKBONES


def get_args():
    parser = argparse.ArgumentParser(description="Parse model training options.")

    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        choices=list(BACKBONES),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="sessions",
    )

    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument(
        "--csv_dir", type=str, default=str(Path(__file__).resolve().parent / "data/csv")
    )

    parser.add_argument("--epochs", type=int, default=100)

    parser.add_argument("--batch_size", type=int, default=32, choices=[16, 32, 64, 128])

    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--weight_decay", type=float, default=1e-2)

    # Epochs without subject-level validation improvement before stopping.
    parser.add_argument("--patience", type=int, default=10)

    # The learning rate is divided by 10 after every this many epochs without improvement.
    parser.add_argument("--lr_patience", type=int, default=3)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_workers", type=int, default=min(8, os.cpu_count() or 1))

    # Random training augmentation (see dataset.TRAIN_AUGMENTATION).
    parser.add_argument(
        "--augment", action=argparse.BooleanOptionalAction, default=True
    )

    # Mixed precision, only used on CUDA.
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)

    return parser.parse_args()
