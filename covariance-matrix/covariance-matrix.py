import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        return None
    N, D = X.shape
    if N < 2:
        return None

    mu = np.mean(X, axis=0)          # (D,)
    Xc = X - mu                      # (N, D)
    
    Sigma = (Xc.T @ Xc) / (N - 1)    # (D, D)

    return Sigma
