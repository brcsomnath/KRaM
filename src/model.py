"""Debiasing Network."""

from torch import nn

class Net(nn.Module):
    """
    Dynamic MLP network with ReLU non-linearity
    """
    def __init__(self, num_layers, size=50, input_size=13):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, size))
        self.layers.append(nn.ReLU())

        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(size, size))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(size, size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class LinearNet(nn.Module):
    """
    Dynamic MLP network with ReLU non-linearity
    """
    def __init__(self, num_layers, size=50, input_size=13):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, size))
        self.layers.append(nn.ReLU())

        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(size, size))

        self.layers.append(nn.Linear(size, size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


