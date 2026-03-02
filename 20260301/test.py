import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

a=np.linspace(-1, 1, 10)
print(a)

b1=torch.linspace(-1, 1, 10)
b2 = torch.unsqueeze(torch.linspace(-1, 1, 10), dim=1)

print(b1)
print(b1.shape)
print(b2)
print(b2.shape)