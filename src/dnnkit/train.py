import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from dnnkit.model import SimpleNet
from dnnkit.data import get_mnist_loaders


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total


def train(epochs=3, lr=1e-3, batch_size=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)

    model = SimpleNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []
    test_accuracies = []

    os.makedirs("outputs", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        acc = evaluate(model, test_loader, device)

        losses.append(avg_loss)
        test_accuracies.append(acc)

        print(f"epoch {epoch + 1}: loss={avg_loss:.4f}, test_acc={acc:.4f}")

    torch.save(model.state_dict(), "outputs/mnist_model.pt")
    print("Saved outputs/mnist_model.pt")

    plt.figure()
    plt.plot(range(1, epochs + 1), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("MNIST Training Loss")
    plt.savefig("outputs/loss_curve.png", dpi=300, bbox_inches="tight")
    print("Saved outputs/loss_curve.png")

    plt.figure()
    plt.plot(range(1, epochs + 1), test_accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("MNIST Test Accuracy")
    plt.savefig("outputs/test_accuracy.png", dpi=300, bbox_inches="tight")
    print("Saved outputs/test_accuracy.png")


if __name__ == "__main__":
    train()
