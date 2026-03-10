from dnnkit.data import get_mnist_loaders
from dnnkit.model import SimpleNet
from dnnkit.models.cnn import SimpleCNN


DATASET_REGISTRY = {
    "mnist": get_mnist_loaders,
}

MODEL_REGISTRY = {
    "mlp": SimpleNet,
    "cnn": SimpleCNN,
}


def get_dataset_loaders(name, batch_size=64):
    name = name.lower()

    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}")

    return DATASET_REGISTRY[name](batch_size=batch_size)


def get_model(name):
    name = name.lower()

    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")

    return MODEL_REGISTRY[name]()
