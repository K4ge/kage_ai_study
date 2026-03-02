import torch.nn as nn

model = nn.Linear(3, 2)

print(model.weight.shape)
print(model.bias.shape)