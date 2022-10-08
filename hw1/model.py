from typing import Tuple, Union
import torch
import torch.nn as nn


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
        activation_fn = torch.nn.LeakyReLU(0.2, inplace=True)
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


class Network(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 10) -> None:
        super().__init__()

        kernel_size: int = 3
        activation: str = "leakyRelu"

        self.main = nn.Sequential(
            # 224 * 224 * in_channels
            *make_convs(
                2,
                kernel_size,
                in_channels,
                64,
                stride=1,
                padding=1,
                bias=True,
                activation=activation,
            ),
            nn.MaxPool2d(2, stride=2),
            # 112 * 112 * 64
            *make_convs(
                2,
                kernel_size,
                64,
                128,
                stride=1,
                padding=1,
                bias=True,
                activation=activation,
            ),
            nn.MaxPool2d(2, stride=2),
            # 56 * 56 * 128
            *make_convs(
                4,
                kernel_size,
                128,
                256,
                stride=1,
                padding=1,
                bias=True,
                activation=activation,
            ),
            nn.MaxPool2d(2, stride=2),
            # 28 * 28 * 256
            *make_convs(
                4,
                kernel_size,
                256,
                512,
                stride=1,
                padding=1,
                bias=True,
                activation=activation,
            ),
            nn.MaxPool2d(2, stride=2),
            # 14 * 14 * 512
            *make_convs(
                4,
                kernel_size,
                512,
                512,
                stride=1,
                padding=1,
                bias=True,
                activation=activation,
            ),
            nn.MaxPool2d(2, stride=2),
            # 7 * 7 * 512
            nn.Flatten(),
            nn.Linear(in_features=7 * 7 * 512, out_features=out_channels),
        )

    @property
    def num_parameters(self) -> int:
        return sum(
            param.numel() for param in self.parameters() if param.requires_grad
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


if __name__ == "__main__":
    inputs = torch.randn((1, 3, 224, 224))
    model = Network(3, 10)
    print(model)
    labels = model(inputs)
    print(labels)
