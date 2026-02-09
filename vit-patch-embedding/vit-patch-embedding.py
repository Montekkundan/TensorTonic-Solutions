import numpy as np

def patch_embed(image: np.ndarray, patch_size: int, embed_dim: int) -> np.ndarray:
    """
    Convert image to patch embeddings.
    """
    if image.ndim != 4:
        raise ValueError(f"Expected image with shape (B,H,W,C), got {image.shape}")

    B, H, W, C = image.shape
    P = patch_size

    if H % P != 0 or W % P != 0:
        raise ValueError("Patch size must evenly divide image height and width.")

    h_patches = H // P
    w_patches = W // P
    N = h_patches * w_patches

    x = image.reshape(B, h_patches, P, w_patches, P, C)

    x = x.transpose(0, 1, 3, 2, 4, 5)

    patch_dim = P * P * C
    patches = x.reshape(B, N, patch_dim)

    rng = np.random.default_rng(0)
    W_proj = rng.standard_normal((patch_dim, embed_dim), dtype=np.float32) / np.sqrt(patch_dim)
    b_proj = np.zeros((embed_dim,), dtype=np.float32)

    embeddings = patches.astype(np.float32) @ W_proj + b_proj  # (B, N, D)
    return embeddings