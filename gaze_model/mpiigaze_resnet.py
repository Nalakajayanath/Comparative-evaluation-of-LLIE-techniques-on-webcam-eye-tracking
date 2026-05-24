# ResNet-preact for MPIIGaze eye patches (hysts/pytorch_mpiigaze, MIT).

import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out")
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.zeros_(module.bias)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                "conv",
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False
                ),
            )

    def forward(self, x):
        x = F.relu(self.bn1(x), inplace=True)
        y = self.conv1(x)
        y = F.relu(self.bn2(y), inplace=True)
        y = self.conv2(y)
        return y + self.shortcut(x)


class MPIIGazeResNetPreact(nn.Module):
    def __init__(self):
        super().__init__()
        n_blocks = 1
        n_channels = [16, 32, 64]

        self.conv = nn.Conv2d(1, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.stage1 = self._make_stage(n_channels[0], n_channels[0], n_blocks, stride=1)
        self.stage2 = self._make_stage(n_channels[0], n_channels[1], n_blocks, stride=2)
        self.stage3 = self._make_stage(n_channels[1], n_channels[2], n_blocks, stride=2)
        self.bn = nn.BatchNorm2d(n_channels[2])

        with torch.no_grad():
            dummy = torch.zeros(1, 1, 36, 60)
            self.feature_size = self._forward_conv(dummy).view(-1).size(0)

        self.fc = nn.Linear(self.feature_size + 2, 2)
        self.apply(initialize_weights)

    @staticmethod
    def _make_stage(in_ch, out_ch, n_blocks, stride):
        blocks = []
        for i in range(n_blocks):
            blocks.append(BasicBlock(out_ch if i else in_ch, out_ch, stride if i == 0 else 1))
            in_ch = out_ch
        return nn.Sequential(*blocks)

    def _forward_conv(self, x):
        x = self.conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.relu(self.bn(x), inplace=True)
        return F.adaptive_avg_pool2d(x, 1)

    def forward(self, x, pose):
        x = self._forward_conv(x).view(x.size(0), -1)
        return self.fc(torch.cat([x, pose], dim=1))
