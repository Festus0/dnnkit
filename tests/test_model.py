import torch
from dnnkit.model import SimpleNet

def test_forward():
    model = SimpleNet()
    x = torch.randn(2, 1, 28, 28)
    y = model(x)
    assert y.shape == (2, 10)
