import torch
import torch.nn as nn
from .conv_layer import ConvLayer  

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, pool_proj):
        super(InceptionModule, self).__init__()

        # 1x1 conv branch
        self.branch1 = ConvLayer(in_channels, out_1x1, kernel_size=1)

        # 1x1 reduction + 3x3 conv branch
        self.branch2 = nn.Sequential(
            ConvLayer(in_channels, red_3x3, kernel_size=1),  # dimension reduction
            ConvLayer(red_3x3, out_3x3, kernel_size=3, padding=1)
        )

        # 1x1 reduction + 5x5 conv branch
        self.branch3 = nn.Sequential(
            ConvLayer(in_channels, red_5x5, kernel_size=1),  # dimension reduction
            ConvLayer(red_5x5, out_5x5, kernel_size=5, padding=2)
        )

        # pooling + 1x1 conv projection branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvLayer(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)
