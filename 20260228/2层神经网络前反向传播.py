import numpy as np

# ========== 1) 数据：1个样本、2个特征 ==========
X = np.array([[1.0, 2.0]])   # (1,2)
y = np.array([[20.0]])       # (1,1)

# ========== 2) 参数 ==========
W1 = np.array([[ 1.0, -1.0],   # (2,2)  第一层：2输入 -> 2隐藏
               [ 2.0,  0.0]])
b1 = np.array([[0.0, 0.0]])    # (1,2)

W2 = np.array([[3.0],          # (2,1)  第二层：2隐藏 -> 1输出
               [4.0]])
b2 = np.array([[0.0]])         # (1,1)

# ========== 3) 激活函数 ==========
def relu(z):
    return np.maximum(0, z)

# ReLU 导数：z>0 为1，否则0
def relu_grad(z):
    return (z > 0).astype(float)

# ========== 4) 前向传播 ==========
Z1 = X @ W1 + b1         # (1,2)
A1 = relu(Z1)            # (1,2)
Z2 = A1 @ W2 + b2        # (1,1)
y_pred = Z2              # 回归：y_pred = Z2

loss = np.mean((y_pred - y)**2)

print("Z1:", Z1, "shape", Z1.shape)
print("A1:", A1, "shape", A1.shape)
print("Z2:", Z2, "shape", Z2.shape)
print("loss:", loss)

# ========== 5) 反向传播（严格按链条） ==========
n = X.shape[0]  # 样本数，这里是1

# (a) dL/dZ2  （因为 y_pred = Z2）
dZ2 = (2/n) * (Z2 - y)           # (1,1)

# (b) 第二层参数梯度
dW2 = A1.T @ dZ2                  # (2,1)
db2 = np.sum(dZ2, axis=0, keepdims=True)  # (1,1)

# (c) 关键：把误差传回到 A1
dA1 = dZ2 @ W2.T                  # (1,2)

# (d) 过 ReLU：A1 = ReLU(Z1)
dZ1 = dA1 * relu_grad(Z1)         # (1,2) 逐元素乘

# (e) 第一层参数梯度
dW1 = X.T @ dZ1                   # (2,2)
db1 = np.sum(dZ1, axis=0, keepdims=True)  # (1,2)

print("\n--- grads ---")
print("dZ2:", dZ2, "shape", dZ2.shape)
print("dW2:", dW2, "shape", dW2.shape)
print("dA1:", dA1, "shape", dA1.shape)
print("dZ1:", dZ1, "shape", dZ1.shape)
print("dW1:", dW1, "shape", dW1.shape)
print("db1:", db1, "shape", db1.shape)