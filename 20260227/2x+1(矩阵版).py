import numpy as np

# 数据
X = np.array([[1],
              [2],
              [3],
              [4]])

y = np.array([[3],
              [5],
              [7],
              [9]])

# 初始化参数
W = np.array([[0.0]])
b = np.array([[0.0]])

lr = 0.01

for epoch in range(200):

    # 前向传播
    y_pred = X @ W + b      # 矩阵乘法

    # 误差
    error = y_pred - y

    # 损失
    loss = np.mean(error ** 2)

    # 梯度（矩阵版）
    dW = (2 / len(X)) * X.T @ error
    db = (2 / len(X)) * np.sum(error)

    # 更新
    W -= lr * dW
    b -= lr * db

    print(f"第{epoch}轮:")
    print("loss=", loss)
    print("W=", W)
    print("b=", b)
    print("---------")

print("最终结果:")
print("W=", W)
print("b=", b)