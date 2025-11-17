import torch.nn as nn

class AvgPoolLayer(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1):
        super(AvgPoolLayer, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.pool(x)
