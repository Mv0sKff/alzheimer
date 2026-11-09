import argparse
import os
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(description="Parse model training options.")

    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50"],
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

    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # Epochs without validation balanced accuracy improvement before stopping.
    parser.add_argument("--patience", type=int, default=10)

    # Epochs without improvement before the learning rate is divided by 10.
    parser.add_argument("--lr_patience", type=int, default=3)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_workers", type=int, default=min(8, os.cpu_count() or 1))

    # Mixed precision, only used on CUDA.
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)

    return parser.parse_args()
