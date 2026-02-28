import numpy as np

# ========== 1) 造数据：y = x1 + x2 ==========
np.random.seed(0)
N = 200  # 样本数
X = np.random.randn(N, 2)              # (N,2)
y = (X[:, [0]] + X[:, [1]])            # (N,1)  注意用 [:,[...]] 保持二维

# ========== 2) 初始化参数（2 -> hidden -> 1）==========
H = 4  # 隐藏层神经元个数（你可以改大/改小）
W1 = 0.1 * np.random.randn(2, H)       # (2,H)
b1 = np.zeros((1, H))                  # (1,H)
W2 = 0.1 * np.random.randn(H, 1)       # (H,1)
b2 = np.zeros((1, 1))                  # (1,1)

lr = 0.05
epochs = 2000

def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(float)

# ========== 3) 训练循环 ==========
for epoch in range(1, epochs + 1):

    # ----- forward -----
    Z1 = X @ W1 + b1          # (N,H)
    A1 = relu(Z1)             # (N,H)
    Z2 = A1 @ W2 + b2         # (N,1)
    y_pred = Z2               # 回归：y_pred = Z2

    # MSE loss
    loss = np.mean((y_pred - y) ** 2)

    # ----- backward（严格按链条） -----
    n = X.shape[0]
    dZ2 = (2 / n) * (y_pred - y)        # (N,1)  = dL/dZ2

    dW2 = A1.T @ dZ2                    # (H,1)
    db2 = np.sum(dZ2, axis=0, keepdims=True)  # (1,1)

    dA1 = dZ2 @ W2.T                    # (N,H)
    dZ1 = dA1 * relu_grad(Z1)           # (N,H)

    dW1 = X.T @ dZ1                     # (2,H)
    db1 = np.sum(dZ1, axis=0, keepdims=True)  # (1,H)

    # ----- update -----
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    # ----- log -----
    if epoch % 200 == 0 or epoch == 1:
        print(f"epoch {epoch:4d} | loss = {loss:.6f}")

# ========== 4) 看看学得怎么样 ==========
test = np.array([[3.0, 5.0],
                 [-2.0, 4.0],
                 [0.1, -0.2]])  # (3,2)
pred = relu(test @ W1 + b1) @ W2 + b2

print("\nTest X:")
print(test)
print("Pred y:")
print(pred)
print("True y:")
print(test[:, [0]] + test[:, [1]])