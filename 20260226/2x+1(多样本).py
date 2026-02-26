# 初始化参数
w = 0.0
b = 0.0
lr = 0.01

# 数据集
x_list = [1, 2, 3, 4]
y_list = [3, 5, 7, 9]

for epoch in range(20):

    total_loss = 0
    total_dw = 0
    total_db = 0

    # 遍历所有样本
    for x, y_true in zip(x_list, y_list):

        # 前向计算
        y_pred = w * x + b

        # 损失
        loss = (y_pred - y_true) ** 2
        total_loss += loss

        # 梯度
        dw = 2 * (y_pred - y_true) * x
        db = 2 * (y_pred - y_true)

        total_dw += dw
        total_db += db

    # 求平均梯度
    avg_dw = total_dw / len(x_list)
    avg_db = total_db / len(x_list)

    # 更新参数
    w -= lr * avg_dw
    b -= lr * avg_db

    print(f"第{epoch}轮:")
    print(f"  平均loss = {total_loss/len(x_list):.4f}")
    print(f"  w = {w:.4f}, b = {b:.4f}")
    print("-----------")

print("最终结果:", w, b)