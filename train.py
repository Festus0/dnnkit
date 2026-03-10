import torch
import matplotlib.pyplot as plt
from dnnkit.model import SimpleNet

model = SimpleNet()
x = torch.randn(100,10)
y = torch.randint(0,2,(100,))

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

losses = []

for epoch in range(10):
    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print(epoch, loss.item())

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("loss_curve.png", dpi=300)
