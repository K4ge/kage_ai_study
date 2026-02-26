
w = 0.0
b = 0.0
lr = 0.01

for i in range(20):   # 先跑20次，方便观察
    # 一组训练数据，x=3，y=7
    x = 3
    y_true = 7   # 2*3 + 1

    # 1️⃣ 前向计算
    y_pred = w * x + b

    # 2️⃣ 计算损失
    loss = (y_pred - y_true) ** 2

    # 3️⃣ 计算梯度（手推导）通过链式求导求出dw和de
    dw = 2 * (y_pred - y_true) * x
    db = 2 * (y_pred - y_true)

    # 4️⃣ 更新参数
    w = w - lr * dw
    b = b - lr * db

    print(f"第{i}次:")
    print(f"  预测值 y_pred={y_pred:.4f}")
    print(f"  loss={loss:.4f}")
    print(f"  w={w:.4f}, b={b:.4f}")
    print("------------")

print("最终结果:", w, b)