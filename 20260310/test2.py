import torch

losses = torch.tensor([1.0, 2.0, 3.0, 4.0])

print(losses)
print(losses.mean().item())