import torch

a = torch.tensor([1, 2, 3])
b = torch.zeros((2, 3))
c = torch.linspace(-1, 1, 5)

print("a:", a, a.shape)
print("b:", b, b.shape)
print("c:", c, c.shape)

d = c.unsqueeze(0)
print("unsqueeze(0):", d, d.shape)

e = c.unsqueeze(1)
print("unsqueeze(1):", e, e.shape)