import numpy as np

def classification_head(encoder_output: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Classification head for ViT.
    """
    if encoder_output.ndim != 3:
        raise ValueError(f"Expected encoder_output with shape (B,N,D), got {encoder_output.shape}")

    B, N, D = encoder_output.shape
    x = encoder_output.astype(np.float32)

    cls = x[:, 0, :]

    eps = 1e-5
    mean = cls.mean(axis=-1, keepdims=True)
    var = ((cls - mean) ** 2).mean(axis=-1, keepdims=True)
    cls_ln = (cls - mean) / np.sqrt(var + eps)

    # (B,D) @ (D,C) -> (B,C)
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((D, num_classes)).astype(np.float32) / np.sqrt(D))
    b = np.zeros((num_classes,), dtype=np.float32)

    logits = cls_ln @ W + b
    return logits