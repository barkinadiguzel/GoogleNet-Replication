import torch.nn as nn

class DropoutLayer(nn.Module):
    def __init__(self, p=0.5):
        super(DropoutLayer, self).__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        return self.dropout(x)
