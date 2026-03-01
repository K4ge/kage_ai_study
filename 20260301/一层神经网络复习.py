import numpy as np

N, D = 200, 2
X = np.random.randn(N, D)
y = (X[:, [0]] + X[:, [1]])   # (N,1)

W = np.zeros((D,1))
b = np.zeros((1,1))
lr = 0.1

for epoch in range(1000):
    y_pred = X @ W + b            # (N,1)
    E = y_pred - y                # (N,1)
    loss = np.mean(E**2)

    dY = (2/N) * E                # (N,1)
    dW = X.T @ dY                 # (D,1)
    db = np.sum(dY, axis=0, keepdims=True)  # (1,1)

    W -= lr * dW
    b -= lr * db

    if epoch % 200 == 0:
        print(epoch, loss)