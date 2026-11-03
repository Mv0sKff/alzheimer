import argparse
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

    parser.add_argument("--epochs", type=int, default=100)  # set to 100

    parser.add_argument("--batch_size", type=int, default=32, choices=[16, 32, 64, 128])

    parser.add_argument("--lr", type=float, default=1e-3, choices=[1e-4, 1e-3])

    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, choices=[1e-5, 1e-7]
    )

    return parser.parse_args()
