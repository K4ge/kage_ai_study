import numpy as np

# 1. 造数据（4个样本，2个特征）
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y = np.array([[0],[0],[0],[1]])  # 类似 AND 逻辑

# 2. 初始化参数
np.random.seed(0)
W = np.random.randn(2,1)
b = np.zeros((1,1))

# 3. sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 4. 训练
lr = 0.1

for i in range(1000):

    # forward
    Z = X @ W + b
    a = sigmoid(Z)

    # loss
    loss = -np.mean(y*np.log(a) + (1-y)*np.log(1-a))

    # backward
    dZ = a - y
    dW = X.T @ dZ / len(X)
    db = np.mean(dZ)

    # update
    W -= lr * dW
    b -= lr * db

    if i % 200 == 0:
        print("loss:", loss)

print("\nFinal prediction:")
print(a)