import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np


# 输入数据
X = torch.tensor([
    [1.0,1.0],
    [1.0,2.0],
    [2.0,1.0],
    [4.0,4.0],
    [5.0,4.0],
    [4.0,5.0]
])

# 标签
y = torch.tensor([0,0,0,1,1,1])
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

print("X shape:",X.shape)
print("y shape:",y.shape)
class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2,8)
        self.fc2 = nn.Linear(8,2)

    def forward(self,x):

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x

model = Net()

print(model)

outputs = model(X)

print("outputs shape:",outputs.shape)
print(outputs)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(200):
    # 1) 前向：得到 logits
    for batch_X, batch_y in loader:
        logits = model(batch_X)

        loss = criterion(logits, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


logits = model(X)
pred = torch.argmax(logits, dim=1)
print("Final pred:", pred)
print("True y   :", y)

# 转成 numpy 方便画图
X_np = X.numpy()
y_np = y.numpy()

# 画原始数据点
for i in range(len(X_np)):
    if y_np[i] == 0:
        plt.scatter(X_np[i,0], X_np[i,1], color="red")
    else:
        plt.scatter(X_np[i,0], X_np[i,1], color="blue")


# 画 decision boundary
x1 = np.linspace(0,6,100)
x2 = np.linspace(0,6,100)

xx1, xx2 = np.meshgrid(x1, x2)

grid = np.c_[xx1.ravel(), xx2.ravel()]
grid_tensor = torch.tensor(grid, dtype=torch.float32)

with torch.no_grad():
    logits = model(grid_tensor)
    pred = torch.argmax(logits, dim=1)

pred = pred.numpy().reshape(xx1.shape)

plt.contourf(xx1, xx2, pred, alpha=0.3)

plt.title("Decision Boundary")
plt.show()