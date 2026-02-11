import numpy as np

def roc_curve(y_true, y_score):
    """
    Compute ROC curve from binary labels and scores.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same shape")
    if y_true.ndim != 1:
        raise ValueError("y_true and y_score must be 1D arrays")

    y_true = (y_true != 0).astype(np.int8)

    n = y_true.size
    if n == 0:
        return (np.array([0.0]), np.array([0.0]), np.array([np.inf]))

    order = np.lexsort((np.arange(n), -y_score))
    y_score_s = y_score[order]
    y_true_s = y_true[order]

    tp_cum = np.cumsum(y_true_s, dtype=np.int64)
    fp_cum = np.cumsum(1 - y_true_s, dtype=np.int64)

    P = tp_cum[-1]
    N = fp_cum[-1]

    score_change = np.diff(y_score_s) != 0
    distinct_ends = np.flatnonzero(score_change)
    distinct_ends = np.r_[distinct_ends, n - 1]

    tp = tp_cum[distinct_ends]
    fp = fp_cum[distinct_ends]
    thr = y_score_s[distinct_ends]

    if P == 0:
        tpr_core = np.zeros_like(tp, dtype=np.float64)
    else:
        tpr_core = tp.astype(np.float64) / float(P)

    if N == 0:
        fpr_core = np.zeros_like(fp, dtype=np.float64)
    else:
        fpr_core = fp.astype(np.float64) / float(N)

    fpr = np.r_[0.0, fpr_core]
    tpr = np.r_[0.0, tpr_core]
    thresholds = np.r_[np.inf, thr]

    return fpr, tpr, thresholds