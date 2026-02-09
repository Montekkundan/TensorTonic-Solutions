import numpy as np

def silhouette_score(X, labels):
    """
    Compute the mean Silhouette Score for given points and cluster labels.
    X: np.ndarray of shape (n_samples, n_features)
    labels: np.ndarray of shape (n_samples,)
    Returns: float
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)

    if X.ndim != 2 or labels.ndim != 1:
        return 0.0
    n = X.shape[0]
    if n < 2 or labels.shape[0] != n:
        return 0.0

    uniq = np.unique(labels)
    K = uniq.size
    if K < 2:
        return 0.0

    diff = X[:, None, :] - X[None, :, :]          # (n, n, d)
    D = np.sqrt((diff ** 2).sum(axis=2))          # (n, n)
    a = np.zeros(n, dtype=float)
    b = np.full(n, np.inf, dtype=float)

    for c in uniq:
        mask_c = (labels == c)
        idx_c = np.where(mask_c)[0]
        m = idx_c.size

        if m > 1:
            intra = D[np.ix_(idx_c, idx_c)]       # (m, m)
            a[idx_c] = intra.sum(axis=1) / (m - 1)
        else:
            a[idx_c] = 0.0

        for c2 in uniq:
            if c2 == c:
                continue
            mask_c2 = (labels == c2)
            idx_c2 = np.where(mask_c2)[0]
            inter = D[np.ix_(idx_c, idx_c2)]       # (m, size_of_c2)
            mean_inter = inter.mean(axis=1)        # (m,)
            b[idx_c] = np.minimum(b[idx_c], mean_inter)

    denom = np.maximum(a, b)
    s = np.zeros(n, dtype=float)
    valid = denom > 0
    s[valid] = (b[valid] - a[valid]) / denom[valid]

    for c in uniq:
        if np.sum(labels == c) == 1:
            s[labels == c] = 0.0

    return float(s.mean())