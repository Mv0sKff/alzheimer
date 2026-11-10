from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parent
# Statistics used to pretrain the torchvision IMAGENET1K_V1 ResNet weights.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Mild geometric and intensity changes for training. No flips, so the left and
# right hemispheres keep their sides. Empty areas are filled with black background.
TRAIN_AUGMENTATION = v2.Compose(
    [
        v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
        # TODO add more
    ]
)

# DataLoader is already parallel so only 1 OpenCV thread
cv2.setNumThreads(0)


def first_scan(df):
    """Keep only the slices of each subject's first scan (session and mpr number)."""
    scan = df["Path"].str.extract(r"_(MR\d+_mpr-\d+)_", expand=False)
    return df[scan == scan.groupby(df["Subject"]).transform("first")]


class SubjectSampler(Sampler):
    """Each epoch, draws slices_per_subject random slices of every subject from all
    its scans, in random order, so every subject counts equally."""

    def __init__(self, subjects, slices_per_subject, generator):
        _, subject_ids = np.unique(subjects, return_inverse=True)
        self.groups = [
            torch.from_numpy(np.flatnonzero(subject_ids == i))
            for i in range(subject_ids.max() + 1)
        ]
        self.slices_per_subject = slices_per_subject
        self.generator = generator

    def __len__(self):
        return len(self.groups) * self.slices_per_subject

    def __iter__(self):
        k = self.slices_per_subject
        picks = [
            group[torch.randperm(len(group), generator=self.generator)[:k]]
            if len(group) >= k
            else group[torch.randint(len(group), (k,), generator=self.generator)]
            for group in self.groups
        ]
        indices = torch.cat(picks)
        order = torch.randperm(len(indices), generator=self.generator)
        yield from indices[order].tolist()


def read_img(file_path):
    img = cv2.imread(str(ROOT / file_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {ROOT / file_path}")
    return img


class alzheimers_dataset(Dataset):
    def __init__(self, dataset, augment=False):
        self.dataset = dataset
        self.augment = augment
        self.paths = dataset["Path"].to_numpy()
        self.subjects = dataset["Subject"].to_numpy()
        self.labels = dataset["Label"].to_numpy()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = read_img(self.paths[idx])

        img = torch.from_numpy(img).float().div(255).unsqueeze(0)
        if self.augment:
            img = TRAIN_AUGMENTATION(img)

        # Grayscale to a 3-channel tensor normalized like the ImageNet inputs.
        img = (img.expand(3, -1, -1) - IMAGENET_MEAN) / IMAGENET_STD

        return {"image": img, "subject": self.subjects[idx], "target": self.labels[idx]}
