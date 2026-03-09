
import torch
from dnnkit.model import SimpleNet

def test_forward():
    model = SimpleNet()
    x = torch.randn(2,10)
    y = model(x)
    assert y.shape[0] == 2
