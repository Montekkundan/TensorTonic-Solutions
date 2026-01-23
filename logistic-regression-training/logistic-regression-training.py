import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    N = X.shape[0]
    D = X.shape[1]
    w = np.zeros(D)
    b = 0.0
    
    for step in range(steps):
        z = X@w + b
        p = _sigmoid(z)

        error = p -y

        w_grad = X.T @ error / N
        b_grad = error.sum() / N

        w = w - lr * w_grad
        b = b - lr * b_grad
        
    return (w, b)
