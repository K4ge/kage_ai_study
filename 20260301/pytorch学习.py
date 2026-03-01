import torch
import torch.nn as nn
import torch.optim as optim

# 1️⃣ 造数据
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = 2 * x + 1 + 0.2 * torch.rand(x.size())

# 2️⃣ 定义模型
model = nn.Linear(1, 1)

# 3️⃣ 定义损失函数
criterion = nn.MSELoss()

# 4️⃣ 定义优化器
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 5️⃣ 训练
for epoch in range(100):

    # 前向传播
    outputs = model(x)
    loss = criterion(outputs, y)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 6️⃣ 打印学到的参数
for name, param in model.named_parameters():
    print(name, param.data)