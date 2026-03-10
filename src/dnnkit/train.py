import os
import json
import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from dnnkit.registry import get_dataset_loaders, get_model


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


def train(epochs=3, lr=1e-3, batch_size=64, dataset_name="mnist", model_name="SimpleNet"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataset_loaders(dataset_name, batch_size=batch_size)

    model = get_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []
    test_accuracies = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("outputs", f"{dataset_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

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

    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
    print(f"Saved {os.path.join(run_dir, 'model.pt')}")

    plt.figure()
    plt.plot(range(1, epochs + 1), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(f"{dataset_name.upper()} Training Loss")
    plt.savefig(os.path.join(run_dir, "loss_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(range(1, epochs + 1), test_accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title(f"{dataset_name.upper()} Test Accuracy")
    plt.savefig(os.path.join(run_dir, "test_accuracy.png"), dpi=300, bbox_inches="tight")
    plt.close()

    metrics = {
        "dataset": dataset_name,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "final_loss": losses[-1],
        "final_test_accuracy": test_accuracies[-1],
        "device": device,
        "run_dir": run_dir,
    }

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved {os.path.join(run_dir, 'metrics.json')}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--model", type=str, default="SimpleNet")
    args = parser.parse_args()

    train(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, dataset_name=args.dataset, model_name=args.model)
