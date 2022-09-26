﻿import os
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

DEFAULT_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


class SportDataset(Dataset):
    def __init__(
        self,
        img_folder: str,
        label_file: str = None,
        transforms: transforms.Compose = DEFAULT_TRANSFORMS,
    ) -> None:
        super().__init__()
        self._img_folder = img_folder
        self._img_names = os.listdir(img_folder)
        if label_file is not None:
            self._labels = pd.read_csv(label_file)
        self._transforms = transforms

    def _get_img(self, name: str) -> torch.Tensor:
        img = Image.open(Path(self._img_folder) / name)
        return self._transforms(img)

    def __len__(self) -> int:
        return len(self._img_names)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        if hasattr(self, "_labels"):
            img_name, label = self._labels.iloc[index]
        else:
            label = -1
            img_name = self._img_names[index]
        img = self._get_img(img_name)
        return img, label
