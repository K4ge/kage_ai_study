import torch

a = torch.tensor([1, 2, 3])
b = torch.zeros((2, 3))
c = torch.linspace(-1, 1, 5)

print("a:", a, a.shape)
print("b:", b, b.shape)
print("c:", c, c.shape)