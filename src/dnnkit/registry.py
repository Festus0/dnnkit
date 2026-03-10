from dnnkit.data import get_mnist_loaders


DATASET_REGISTRY = {
    "mnist": get_mnist_loaders,
}


def get_dataset_loaders(name: str, batch_size: int = 64):
    name = name.lower()
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[name](batch_size=batch_size)
