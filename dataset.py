from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent
# Statistics used to pretrain the torchvision IMAGENET1K_V1 ResNet weights.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# DataLoader is already parallel so only 1 OpenCV thread
cv2.setNumThreads(0)


def read_img(file_path):
    img = cv2.imread(str(ROOT / file_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {ROOT / file_path}")
    return img


class alzheimers_dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.paths = dataset["Path"].to_numpy()
        self.subjects = dataset["Subject"].to_numpy()
        self.labels = dataset["Label"].to_numpy()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = read_img(self.paths[idx])

        # Grayscale to a 3-channel tensor normalized like the ImageNet inputs.
        img = torch.from_numpy(img).float().div(255).expand(3, -1, -1)
        img = (img - IMAGENET_MEAN) / IMAGENET_STD

        return {"image": img, "subject": self.subjects[idx], "target": self.labels[idx]}
