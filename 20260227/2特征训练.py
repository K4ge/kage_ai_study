import numpy as np

# 数据（2个特征）
X = np.array([
    [1, 1],
    [2, 1],
    [3, 2],
    [4, 3]
])

y = np.array([
    [5],
    [7],
    [12],
    [17]
])

# 初始化参数
W = np.array([
    [0.0],
    [0.0]
])

b = np.array([[0.0]])

lr = 0.01

for epoch in range(5000):

    # 前向
    y_pred = X @ W + b

    # 误差
    error = y_pred - y

    # 损失
    loss = np.mean(error ** 2)

    # 梯度
    dW = (2 / len(X)) * X.T @ error
    db = (2 / len(X)) * np.sum(error)

    # 更新
    W -= lr * dW
    b -= lr * db

    if epoch % 10 == 0:
        print(f"epoch {epoch}")
        print("loss:", loss)
        print("W:", W.flatten())
        print("b:", b)
        print("--------")

print("最终结果")
print("W:", W.flatten())
print("b:", b)