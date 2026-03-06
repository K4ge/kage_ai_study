import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
start_time = time.time()

# 1. 数据
transform = transforms.ToTensor()

train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

# 2. 模型
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
        x = self.conv1(x)      # [B,1,28,28] -> [B,16,26,26]
        x = self.relu1(x)
        x = self.pool1(x)      # [B,16,13,13]

        x = self.conv2(x)      # [B,16,13,13] -> [B,32,11,11]
        x = self.relu2(x)
        x = self.pool2(x)      # [B,32,5,5]

        x = torch.flatten(x, 1)  # [B,32,5,5] -> [B,800]
        x = self.fc(x)           # [B,800] -> [B,10]
        return x

model = MyCNN()

# 3. 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 4. 训练
epochs = 3

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"epoch {epoch+1}, train loss = {avg_loss:.4f}")

# 5. 测试
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

acc = correct / total
print(f"test accuracy = {acc:.4f}")

end_time = time.time()
print("total time:", end_time - start_time, "seconds")