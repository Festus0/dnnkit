
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, input_dim=10, hidden=64, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    model = SimpleNet()
    x = torch.randn(4,10)
    y = model(x)
    print("Output shape:", y.shape)
