import torch
import torch.nn as nn

from utils import make_convs


class Network(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 10) -> None:
        super().__init__()

        kernel_size: int = 3
        activation: str = "relu"

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
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            # nn.Linear(in_features=4096, out_features=4096),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            # nn.Linear(in_features=4096, out_features=out_channels),
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
