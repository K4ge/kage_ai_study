import numpy as np

# 输入
X = np.array([
    [1, 2],
    [3, 4]
])

# 第一层
W1 = np.array([
    [1, -1],
    [2,  0]
])
b1 = np.array([[0, 0]])

# 第二层
W2 = np.array([
    [1],
    [2]
])
b2 = np.array([[0]])

# ReLU函数
def relu(x):
    return np.maximum(0, x)

# 前向传播
Z1 = X @ W1 + b1
A1 = relu(Z1)
Z2 = A1 @ W2 + b2

print("Z1:")
print(Z1)

print("A1:")
print(A1)

print("最终输出:")
print(Z2)