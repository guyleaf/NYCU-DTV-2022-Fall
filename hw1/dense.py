from typing import Tuple, Union
from collections import OrderedDict

import torch
import torch.nn as nn


def build_depthwise_separable_conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    bias: bool = False,
    pre_activation: bool = False,
) -> nn.Sequential:
    layers = []

    if pre_activation:
        layers.append(("norm", nn.BatchNorm2d(in_channels)))
        layers.append(("relu", nn.ReLU(inplace=True)))

    layers.append(
        (
            "depthwise_conv",
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=bias,
            ),
        )
    )

    # if pre_activation:
    #     layers.append(("pointwise_norm", nn.BatchNorm2d(in_channels)))
    #     layers.append(("pointwise_relu", nn.ReLU(inplace=True)))
    # else:
    #     layers.append(("depthwise_norm", nn.BatchNorm2d(in_channels)))
    #     layers.append(("depthwise_relu", nn.ReLU(inplace=True)))

    layers.append(
        (
            "pointwise_conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=bias,
            ),
        )
    )

    if not pre_activation:
        layers.append(("norm", nn.BatchNorm2d(out_channels)))
        layers.append(("relu", nn.ReLU(inplace=True)))

    return nn.Sequential(OrderedDict(layers))


# pre-activation: https://arxiv.org/pdf/1707.06990.pdf
def build_dense_layer(
    in_channels: int, growth_rate: int, drop_rate: float
) -> nn.Sequential:
    layers = []
    layers.append(nn.BatchNorm2d(num_features=in_channels))
    layers.append(nn.ReLU(inplace=True))
    layers.append(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=4 * growth_rate,
            kernel_size=1,
            stride=1,
            bias=False,
        )
    )

    in_channels = 4 * growth_rate

    # BatchNorm + ReLU + Conv3x3
    layers.append(
        build_depthwise_separable_conv2d(
            in_channels, growth_rate, 3, 1, 1, pre_activation=True
        )
    )
    # layers.append(nn.BatchNorm2d(in_channels))
    # layers.append(nn.ReLU(inplace=True))
    # layers.append(
    #     nn.Conv2d(
    #         in_channels=in_channels,
    #         out_channels=growth_rate,
    #         kernel_size=3,
    #         stride=1,
    #         padding=1,
    #         bias=False,
    #     )
    # )

    if drop_rate > 0:
        layers.append(nn.Dropout2d(p=drop_rate, inplace=True))
    return nn.Sequential(*layers)


def build_transition_layer(
    in_channels: int, out_channels: int
) -> nn.Sequential:
    layers = []
    layers.append(nn.BatchNorm2d(in_channels))
    layers.append(nn.ReLU(inplace=True))
    layers.append(
        nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, bias=False
        )
    )
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


class DenseBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_layers: int,
        growth_rate: int,
        drop_rate: float = 0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = build_dense_layer(
                in_channels + i * growth_rate, growth_rate, drop_rate
            )
            self.layers.add_module(f"denseLayer{i + 1}", layer)

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = torch.concat([x, layer(x)], dim=1)
        return x
