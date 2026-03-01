import numpy as np

# 1. 造数据（6个样本，2个特征，3个类别）
X = np.array([
    [1,0],
    [0,1],
    [1,1],
    [2,1],
    [1,2],
    [2,2]
])

# 类别：0,1,2
y_labels = np.array([0,1,2,0,1,2])

# 转成 one-hot
num_classes = 3
Y = np.zeros((len(y_labels), num_classes))
Y[np.arange(len(y_labels)), y_labels] = 1

# 2. 初始化参数
np.random.seed(0)
W = np.random.randn(2,3)
b = np.zeros((1,3))

# 3. softmax（带数值稳定）
def softmax(Z):
    Z = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

# 4. 训练
lr = 0.1

for i in range(1000):

    # forward
    Z = X @ W + b
    A = softmax(Z)

    # loss（多分类交叉熵）
    loss = -np.mean(np.sum(Y * np.log(A), axis=1))

    # backward
    dZ = A - Y
    dW = X.T @ dZ / len(X)
    db = np.mean(dZ, axis=0, keepdims=True)

    # update
    W -= lr * dW
    b -= lr * db

    if i % 200 == 0:
        print("loss:", loss)

print("\nFinal probabilities:")
print(A)

print("\nPredicted class:")
print(np.argmax(A, axis=1))