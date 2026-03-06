import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, 3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.fc = nn.Linear(32 * 5 * 5, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


model = MyCNN()
x = torch.randn(4, 1, 28, 28)
y = model(x)

print("input shape:", x.shape)
print("output shape:", y.shape)

# 1) 把图片转成 tensor
transform = transforms.ToTensor()

# 2) 加载 MNIST
train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# 3) DataLoader
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

print("train size:", len(train_ds))
print("test size:", len(test_ds))

images, labels = next(iter(train_loader))
print("images shape:", images.shape)
print("labels shape:", labels.shape)
print("labels sample:", labels[:10])

model = MyCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for images, labels in train_loader:
    outputs = model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("loss:", loss.item())
    break