import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Apply position-wise feed-forward network.
    """
    hidden = np.dot(x, W1) + b1
    relu_out = np.maximum(0, hidden)
    output = np.dot(relu_out, W2) + b2
    return output