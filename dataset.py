import cv2
import torch
from torch.utils.data import Dataset


def read_img(file_path):
    xray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return xray


class knee_xray_dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = read_img(self.dataset["Path"].iloc[idx])

        # work around: Grayscale to normalized 3-channel tensor.
        img = torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1).float().div(255)

        name = self.dataset["Name"].iloc[idx]
        target = self.dataset["KL"].iloc[idx]

        result = {"xray": img, "name": name, "target": target}

        return result
