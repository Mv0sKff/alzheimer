from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent


def read_img(file_path):
    return cv2.imread(str(ROOT / file_path), cv2.IMREAD_GRAYSCALE)


class alzheimers_dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = read_img(self.dataset["Path"].iloc[idx])

        # work around: Grayscale to normalized 3-channel tensor.
        img = torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1).float().div(255)

        subject = self.dataset["Subject"].iloc[idx]
        target = self.dataset["Label"].iloc[idx]

        return {"image": img, "subject": subject, "target": target}
