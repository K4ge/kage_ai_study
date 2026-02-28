import numpy as np

# 数据
X = np.array([
    [1, 2],
    [3, 4]
])

y = np.array([
    [3],
    [7]
])

# 初始化参数
np.random.seed(0)

W1 = np.random.randn(2,2)
b1 = np.zeros((1,2))

W2 = np.random.randn(2,1)
b2 = np.zeros((1,1))

def relu(x):
    return np.maximum(0, x)

# 前向传播
Z1 = X @ W1 + b1
A1 = relu(Z1)
Z2 = A1 @ W2 + b2

y_pred = Z2

loss = np.mean((y_pred - y)**2)

print("预测值:")
print(y_pred)

print("loss:")
print(loss)