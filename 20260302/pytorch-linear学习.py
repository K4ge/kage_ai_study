import torch
import torch.nn as nn

model = nn.Linear(1, 1)

print("weight:", model.weight)
print("bias:", model.bias)

x = torch.tensor([[2.0]])

output = model(x)

print("model output:", output)