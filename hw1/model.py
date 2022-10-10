from typing import Tuple

import torch
import torch.nn as nn

from dense import (
    DenseBlock,
    build_depthwise_separable_conv2d,
    build_transition_layer,
)


class Network(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 10) -> None:
        super().__init__()

        block_config: Tuple[int, ...] = (6, 12, 24, 16)
        growth_rate: int = 32
        drop_rate: float = 0.5

        first_out_channels = 2 * growth_rate
        self.firstBlock = nn.Sequential(
            build_depthwise_separable_conv2d(
                in_channels, first_out_channels, 7, 2, 3, pre_activation=False
            ),
            # nn.Conv2d(
            #     in_channels=in_channels,
            #     out_channels=first_out_channels,
            #     kernel_size=7,
            #     stride=2,
            #     padding=3,
            #     bias=False,
            # ),
            # nn.BatchNorm2d(first_out_channels),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        in_channels = first_out_channels
        self.blocks = nn.Sequential()
        for i, num_blocks in enumerate(block_config):
            block = DenseBlock(
                in_channels=in_channels,
                num_layers=num_blocks,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
            )
            self.blocks.add_module(f"denseBlock{i + 1}", block)
            in_channels = in_channels + num_blocks * growth_rate
            if i != len(block_config) - 1:
                self.blocks.add_module(
                    f"transitionLayer{i + 1}",
                    build_transition_layer(in_channels, in_channels // 2),
                )
                in_channels = in_channels // 2

        self.blocks.append(nn.BatchNorm2d(in_channels))
        self.blocks.append(nn.ReLU(inplace=True))
        self.blocks.append(nn.AdaptiveMaxPool2d(1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_channels, out_features=out_channels),
        )

    @torch.jit.unused
    @property
    def num_parameters(self) -> int:
        return sum(
            param.numel() for param in self.parameters() if param.requires_grad
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.firstBlock(x)
        y = self.blocks(x)
        y = self.classifier(y)
        return y


if __name__ == "__main__":
    inputs = torch.randn((2, 3, 224, 224))
    model = Network(3, 10)
    print(model)
    labels = model(inputs)
    print(labels)
    print(model.num_parameters)
