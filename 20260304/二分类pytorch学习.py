import torch.nn as nn
import torch
import torch.optim as optim

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
    logits = model(X)                 # (6,2)

    # 2) loss：logits + 标签y(6,)
    loss = criterion(logits, y)

    # 3) 反向 + 更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 4) 每隔一段打印一下
    if epoch % 20 == 0:
        # 预测类别：取每行最大值的下标
        pred = torch.argmax(logits, dim=1)
        acc = (pred == y).float().mean().item()
        print(f"Epoch {epoch:3d} | Loss {loss.item():.4f} | Acc {acc:.2f}")

logits = model(X)
pred = torch.argmax(logits, dim=1)
print("Final pred:", pred)
print("True y   :", y)