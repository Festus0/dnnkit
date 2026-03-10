import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, input_dim=28 * 28, hidden=128, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)
