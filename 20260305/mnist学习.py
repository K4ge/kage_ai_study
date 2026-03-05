import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

print("torch:", torch.__version__)

# 1) 把图片转成 Tensor，并归一化到 [0,1]
transform = transforms.ToTensor()

# 2) 下载/读取 MNIST
train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

print("train size:", len(train_ds))
print("test  size:", len(test_ds))

# 3) DataLoader 负责自动吐 batch
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False)

# 4) 取一个 batch 看看长什么样
images, labels = next(iter(train_loader))

print("images shape:", images.shape)   # 期望: (64, 1, 28, 28)
print("labels shape:", labels.shape)   # 期望: (64,)
print("labels sample:", labels[:10])
print("images min/max:", images.min().item(), images.max().item())

img = images[0]
label = labels[0]

plt.imshow(img.squeeze(), cmap="gray")
plt.title(f"label = {label.item()}")
plt.show()