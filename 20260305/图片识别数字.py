import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("cudnn:", torch.backends.cudnn.version())
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

print("torch:", torch.__version__)

# 1) 把图片转成 Tensor，并归一化到 [0,1]
transform = transforms.ToTensor()

# 2) 下载/读取 MNIST
train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

print("train size:", len(train_ds))
print("test  size:", len(test_ds))

# 3) DataLoader 负责自动吐 batch
train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=1024, shuffle=False)

# 4) 取一个 batch 看看长什么样
images, labels = next(iter(train_loader))

print("images shape:", images.shape)   # 期望: (64, 1, 28, 28)
print("labels shape:", labels.shape)   # 期望: (64,)
print("labels sample:", labels[:10])
print("images min/max:", images.min().item(), images.max().item())

img = images[0]
label = labels[0]
#
# plt.imshow(img.squeeze(), cmap="gray")
# plt.title(f"label = {label.item()}")
# plt.show()

class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784,128)
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):

        x = x.view(x.size(0), -1)  # flatten

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x

model = Net().to(device)

images = images.to(device)
labels = labels.to(device)
out = model(images)

print("output shape:", out.shape)
pred = torch.argmax(out, dim=1)

print("pred:", pred[:10])
print("true:", labels[:10])

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def evaluate(model, loader):
    model.eval()  # 进入评估模式（先记住：评估时用它）
    correct = 0
    total = 0

    with torch.no_grad():  # 评估不需要梯度，省内存更快
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)              # (batch,10)
            pred = torch.argmax(logits, dim=1)  # (batch,)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    model.train()  # 切回训练模式
    return correct / total

epochs = 3

for epoch in range(epochs):
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        # 1) forward
        logits = model(images)

        # 2) loss
        loss = criterion(logits, labels)

        # 3) backward + update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 每个 epoch 结束测一次测试准确率
    test_acc = evaluate(model, test_loader)
    print(f"Epoch {epoch+1}/{epochs} | loss={loss.item():.4f} | test_acc={test_acc:.4f}")