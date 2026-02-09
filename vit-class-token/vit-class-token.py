import numpy as np

def prepend_class_token(patches: np.ndarray, embed_dim: int) -> np.ndarray:
    """
    Prepend learnable [CLS] token to patch sequence.
    """
    if patches.ndim != 3:
        raise ValueError(f"Expected patches with shape (B,N,D), got {patches.shape}")

    B, N, D = patches.shape
    if D != embed_dim:
        raise ValueError(f"embed_dim mismatch: patches has D={D}, but embed_dim={embed_dim}")

    rng = np.random.default_rng(0)
    cls_token = rng.normal(loc=0.0, scale=0.02, size=(1, 1, D)).astype(np.float32)  # (1,1,D)

    # (B,1,D)
    cls_batch = np.tile(cls_token, (B, 1, 1))

    out = np.concatenate([cls_batch, patches.astype(np.float32)], axis=1)  # (B, N+1, D)
    return out