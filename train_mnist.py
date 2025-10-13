# train_mnist.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -----------------------
# Define CNN Model
# -----------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # input: 28x28 -> 26x26
        self.conv2 = nn.Conv2d(32, 64, 3, 1) # 26x26 -> 24x24

        # Compute flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1,1,28,28)
            dummy = torch.relu(self.conv1(dummy))
            dummy = torch.relu(self.conv2(dummy))
            flattened_size = dummy.numel()  # 64*24*24 = 36864

        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -----------------------
# Prepare Data
# -----------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# -----------------------
# Train Model
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

epochs = 10  # increase epochs for better accuracy
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    model.train()
    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
    # Validation accuracy after each epoch
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = 100. * correct / total
    print(f"Validation Accuracy after epoch {epoch+1}: {acc:.2f}%")

# Save the trained model
torch.jit.script(model).save("mnist_cnn.pt")
print("Model saved as mnist_cnn.pt")
