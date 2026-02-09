import numpy as np

def add_position_embedding(patches: np.ndarray, num_patches: int, embed_dim: int) -> np.ndarray:
    """
    Add learnable position embeddings to patch embeddings.
    """
    if patches.ndim != 3:
        raise ValueError(f"Expected patches with shape (B,N,D), got {patches.shape}")

    B, N, D = patches.shape

    if N != num_patches:
        raise ValueError(f"num_patches mismatch: patches has N={N}, but num_patches={num_patches}")
    if D != embed_dim:
        raise ValueError(f"embed_dim mismatch: patches has D={D}, but embed_dim={embed_dim}")

    sigma = 0.02
    rng = np.random.default_rng(0)
    pos_table = rng.normal(loc=0.0, scale=sigma, size=(N, D)).astype(np.float32)  # (N, D)

    pos_embed = pos_table[None, :, :]  # (1, N, D)

    out = patches.astype(np.float32) + pos_embed  # (B, N, D)
    return out