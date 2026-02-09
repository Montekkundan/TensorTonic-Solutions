import numpy as np

def vit_encoder_block(x: np.ndarray, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0) -> np.ndarray:
    """
    ViT Transformer encoder block.
    """
    # Architecture:
    # x' = x + MSA(LN(x))
    # x'' = x' + MLP(LN(x'))

    if x.ndim != 3:
        raise ValueError(f"Expected x with shape (B,N,D), got {x.shape}")
    B, N, D = x.shape
    if D != embed_dim:
        raise ValueError(f"embed_dim mismatch: x has D={D}, but embed_dim={embed_dim}")
    if embed_dim % num_heads != 0:
        raise ValueError("embed_dim must be divisible by num_heads.")
    head_dim = embed_dim // num_heads
    hidden_dim = int(embed_dim * mlp_ratio)

    # ----- helpers -----
    def layer_norm(t: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mean = t.mean(axis=-1, keepdims=True)
        var = ((t - mean) ** 2).mean(axis=-1, keepdims=True)
        return (t - mean) / np.sqrt(var + eps)

    def gelu(u: np.ndarray) -> np.ndarray:
        return 0.5 * u * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (u + 0.044715 * (u ** 3))))

    def softmax(a: np.ndarray, axis: int = -1) -> np.ndarray:
        a = a - a.max(axis=axis, keepdims=True)
        ea = np.exp(a)
        return ea / ea.sum(axis=axis, keepdims=True)

    rng = np.random.default_rng(0)
    def init_w(shape, scale=None):
        fan_in = shape[0]
        s = (1.0 / np.sqrt(fan_in)) if scale is None else scale
        return (rng.standard_normal(shape).astype(np.float32) * s)

    x = x.astype(np.float32)

    # =========================
    # 1) x' = x + MSA(LN(x))
    # =========================
    x_ln = layer_norm(x)

    # QKV: (B,N,D) @ (D,3D) -> (B,N,3D)
    W_qkv = init_w((D, 3 * D))
    b_qkv = np.zeros((3 * D,), dtype=np.float32)
    qkv = x_ln @ W_qkv + b_qkv

    # (B,N,D) -> (B,h,N,head_dim)
    q, k, v = np.split(qkv, 3, axis=-1)
    q = q.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)

    # attention scores: (B,h,N,head_dim) x (B,h,head_dim,N) -> (B,h,N,N)
    scale = 1.0 / np.sqrt(head_dim)
    attn_scores = (q @ k.transpose(0, 1, 3, 2)) * scale
    attn = softmax(attn_scores, axis=-1)

    # weighted sum: (B,h,N,N) @ (B,h,N,head_dim) -> (B,h,N,head_dim)
    out_heads = attn @ v

    # merge heads: (B,h,N,head_dim) -> (B,N,D)
    out = out_heads.transpose(0, 2, 1, 3).reshape(B, N, D)

    # output projection: (B,N,D) @ (D,D) -> (B,N,D)
    W_o = init_w((D, D))
    b_o = np.zeros((D,), dtype=np.float32)
    msa_out = out @ W_o + b_o

    x_prime = x + msa_out

    # =========================
    # 2) x'' = x' + MLP(LN(x'))
    # =========================
    x_prime_ln = layer_norm(x_prime)

    W1 = init_w((D, hidden_dim))
    b1 = np.zeros((hidden_dim,), dtype=np.float32)
    W2 = init_w((hidden_dim, D))
    b2 = np.zeros((D,), dtype=np.float32)

    mlp_hidden = gelu(x_prime_ln @ W1 + b1)     # (B,N,4D)
    mlp_out = mlp_hidden @ W2 + b2              # (B,N,D)

    x_out = x_prime + mlp_out
    return x_out