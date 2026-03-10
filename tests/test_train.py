import torch
from dnnkit.model import SimpleNet

def test_model_output_shape():
    model = SimpleNet()
    x = torch.randn(4, 1, 28, 28)
    y = model(x)
    assert y.shape == (4, 10)
