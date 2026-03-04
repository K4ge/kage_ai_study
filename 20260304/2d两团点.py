import torch
import torch.nn as nn

# 让每次运行结果一致
torch.manual_seed(0)

# ======================
# 1) 造数据：两团 2D 点（二分类）
# ======================
n = 400  # 总样本数
x0 = torch.randn(n//2, 2) * 0.8 + torch.tensor([-2.0, 0.0])  # 类0
x1 = torch.randn(n//2, 2) * 0.8 + torch.tensor([ 2.0, 0.0])  # 类1

X = torch.cat([x0, x1], dim=0)                 # (n, 2)
y = torch.cat([torch.zeros(n//2), torch.ones(n//2)]).long()  # (n,) 取值0/1

# 打乱
perm = torch.randperm(n)
X, y = X[perm], y[perm]

# 简单切分训练/测试
split = int(0.8 * n)
X_train, y_train = X[:split], y[:split]
X_test,  y_test  = X[split:], y[split:]


# ======================
# 2) 自定义模型：nn.Module + forward
# ======================
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 2)   # 二分类 => 输出2个logits
        )

    def forward(self, x):
        return self.net(x)   # (batch, 2)

model = Net()

# ======================
# 3) 损失 & 优化器
# ======================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# ======================
# 4) 训练循环（先不用 DataLoader，手写 batch）
# ======================
batch_size = 32
epochs = 30

for epoch in range(1, epochs + 1):
    model.train()

    # 每个epoch都打乱训练集
    perm = torch.randperm(len(X_train))
    X_train_shuf = X_train[perm]
    y_train_shuf = y_train[perm]

    total_loss = 0.0
    correct = 0
    total = 0

    # 按batch切片
    for i in range(0, len(X_train_shuf), batch_size):
        xb = X_train_shuf[i:i+batch_size]  # (B, 2)
        yb = y_train_shuf[i:i+batch_size]  # (B,)

        logits = model(xb)                 # (B, 2)
        loss = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)

        pred = logits.argmax(dim=1)        # (B,)
        correct += (pred == yb).sum().item()
        total += xb.size(0)

    train_loss = total_loss / total
    train_acc = correct / total

    # 每个epoch顺便测一下
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test)
        test_pred = test_logits.argmax(dim=1)
        test_acc = (test_pred == y_test).float().mean().item()

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:02d} | loss={train_loss:.4f} | train_acc={train_acc:.3f} | test_acc={test_acc:.3f}")