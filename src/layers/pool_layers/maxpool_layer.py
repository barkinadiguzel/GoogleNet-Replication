import torch.nn as nn

class MaxPoolLayer(nn.Module):
    def __init__(self, kernel_size=3, stride=2, padding=1):
        super(MaxPoolLayer, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.pool(x)
