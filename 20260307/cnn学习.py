import torch
import torch.nn as nn

# 随机生成一张“假图片”
x = torch.randn(1, 1, 28, 28)
print("input:", x.shape)

# 卷积层
conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3,padding=1)
x = conv(x)
print("after conv:", x.shape)

# ReLU
relu = nn.ReLU()
x = relu(x)
print("after relu:", x.shape)

# MaxPooling
pool = nn.MaxPool2d(kernel_size=2)
x = pool(x)
print("after pool:", x.shape)