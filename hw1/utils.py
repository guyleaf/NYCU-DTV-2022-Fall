import random
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
import matplotlib.pyplot as plt

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


def show_curves(path: str, metrics: dict, every_num_epochs_for_val: int):
    train_loss = metrics["train_loss"]
    val_loss = metrics["val_loss"]
    train_acc = metrics["train_acc"]
    val_acc = metrics["val_acc"]

    train_epochs = list(range(1, len(train_loss) + 1))
    val_epochs = list(range(1, len(train_loss) + 1, every_num_epochs_for_val))

    plt.figure(fontsize=14)
    plt.title("loss curve")
    plt.plot(train_epochs, train_loss, "ro", label="train loss")
    plt.plot(val_epochs, val_loss, "bo", label="validation loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(Path(path) / "loss.png")

    plt.figure(fontsize=14)
    plt.title("accuracy curve")
    epochs = list(range(1, len(train_loss) + 1))
    plt.plot(epochs, train_acc, "ro", label="train accuracy")
    epochs = list(range(1, len(train_loss) + 1, every_num_epochs_for_val))
    plt.plot(epochs, val_acc, "bo", label="validation accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(Path(path) / "accuracy.png")
