import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.bbb = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.bbb(x)
        return x

model = Net()

x = torch.randn(3, 2)   # 3个样本，每个2维特征
y = model(x)

print("x shape:", x.shape)
print("y shape:", y.shape)
print(model.parameters())