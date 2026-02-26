import numpy as np

# 初始化参数
w = 0.0
b = 0.0
lr = 0.01

# 数据（变成 numpy 数组）
x = np.array([1, 2, 3, 4])
y = np.array([3, 5, 7, 9])

for epoch in range(20):

    # 前向计算（向量化）
    y_pred = w * x + b

    # 计算误差
    error = y_pred - y

    # 计算损失
    loss = np.mean(error ** 2)

    # 计算梯度（向量版）
    dw = np.mean(2 * error * x)
    db = np.mean(2 * error)

    # 更新参数
    w -= lr * dw
    b -= lr * db

    print(f"第{epoch}轮:")
    print(f"  loss={loss:.4f}")
    print(f"  w={w:.4f}, b={b:.4f}")
    print("-----------")

print("最终结果:", w, b)