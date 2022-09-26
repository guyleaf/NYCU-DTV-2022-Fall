﻿import random
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch

from dataset import SportDataset


def make_dataset(data_folder: str, mode: str = "train") -> SportDataset:
    img_folder = str(Path(data_folder) / mode)
    label_file = str(Path(data_folder) / f"{mode}.csv")
    return SportDataset(img_folder, label_file)


def make_convs(
    num_layers: int,
    kernel_size: Union[int, Tuple[int, int]],
    in_channels: int,
    out_channels: int,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 1,
    bias: bool = True,
    activation: str = "relu",
) -> list[torch.nn.Module]:
    if activation == "relu":
        activation_fn = torch.nn.ReLU(inplace=True)
    elif activation == "leakyRelu":
        activation_fn = torch.nn.LeakyReLU(inplace=True)
    else:
        raise RuntimeError(
            f"The activation function {activation} is not implemented."
        )

    layers = []
    for _ in range(num_layers):
        layers.append(
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
        )
        layers.append(torch.nn.BatchNorm2d(out_channels))
        layers.append(activation_fn)
        in_channels = out_channels
    return layers


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
