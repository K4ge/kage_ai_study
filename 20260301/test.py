import numpy as np
def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(float)



X= np.array([[1,-2,3],[-1,-2,-3]])

print(relu_grad(X))