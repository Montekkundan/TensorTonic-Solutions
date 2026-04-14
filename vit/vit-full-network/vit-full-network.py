import numpy as np
import math


class VisionTransformer:
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        W_patch=None,
        cls_token=None,
        pos_embed=None,
        encoder_weights=None,
        W_head=None,
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.num_patches = (image_size // patch_size) * (image_size // patch_size)
        self.head_dim = embed_dim // num_heads
        self.hidden_dim = int(embed_dim * mlp_ratio)
        self.eps = 1e-5

        # Patch projection
        if W_patch is not None:
            self.W_patch = np.array(W_patch, dtype=np.float64)
            patch_dim = self.W_patch.shape[0]
            self.in_channels = patch_dim // (patch_size * patch_size)
        else:
            self.in_channels = 3
            patch_dim = patch_size * patch_size * self.in_channels
            self.W_patch = np.random.randn(patch_dim, embed_dim) * 0.02

        # CLS token
        if cls_token is not None:
            self.cls_token = np.array(cls_token, dtype=np.float64)
        else:
            self.cls_token = np.random.randn(1, 1, embed_dim) * 0.02

        # Position embeddings
        if pos_embed is not None:
            self.pos_embed = np.array(pos_embed, dtype=np.float64)
        else:
            self.pos_embed = np.random.randn(1, self.num_patches + 1, embed_dim) * 0.02

        # Encoder weights
        if encoder_weights is not None:
            self.encoder_weights = []
            for layer in encoder_weights:
                self.encoder_weights.append({
                    "Wq": np.array(layer["Wq"], dtype=np.float64),
                    "Wk": np.array(layer["Wk"], dtype=np.float64),
                    "Wv": np.array(layer["Wv"], dtype=np.float64),
                    "Wo": np.array(layer["Wo"], dtype=np.float64),
                    "W1": np.array(layer["W1"], dtype=np.float64),
                    "W2": np.array(layer["W2"], dtype=np.float64),
                })
        else:
            self.encoder_weights = []
            for _ in range(depth):
                self.encoder_weights.append({
                    "Wq": np.random.randn(embed_dim, embed_dim) * 0.02,
                    "Wk": np.random.randn(embed_dim, embed_dim) * 0.02,
                    "Wv": np.random.randn(embed_dim, embed_dim) * 0.02,
                    "Wo": np.random.randn(embed_dim, embed_dim) * 0.02,
                    "W1": np.random.randn(embed_dim, self.hidden_dim) * 0.02,
                    "W2": np.random.randn(self.hidden_dim, embed_dim) * 0.02,
                })

        # Classification head
        if W_head is not None:
            self.W_head = np.array(W_head, dtype=np.float64)
        else:
            self.W_head = np.random.randn(embed_dim, num_classes) * 0.02

    def _layer_norm(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + self.eps)

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        erf_vec = np.vectorize(math.erf)
        return 0.5 * x * (1.0 + erf_vec(x / np.sqrt(2.0)))

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        x = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def _patchify(self, x: np.ndarray) -> np.ndarray:
        B, H, W, C = x.shape
        P = self.patch_size
        h_p = H // P
        w_p = W // P

        patches = x.reshape(B, h_p, P, w_p, P, C)
        patches = patches.transpose(0, 1, 3, 2, 4, 5)
        patches = patches.reshape(B, h_p * w_p, P * P * C)
        return patches

    def _msa(self, x: np.ndarray, weights: dict) -> np.ndarray:
        B, T, D = x.shape
        h = self.num_heads
        d = self.head_dim

        q = x @ weights["Wq"]
        k = x @ weights["Wk"]
        v = x @ weights["Wv"]

        q = q.reshape(B, T, h, d).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, h, d).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, h, d).transpose(0, 2, 1, 3)

        scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(d)
        attn = self._softmax(scores, axis=-1)
        out = attn @ v

        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
        return out @ weights["Wo"]

    def _mlp(self, x: np.ndarray, weights: dict) -> np.ndarray:
        return self._gelu(x @ weights["W1"]) @ weights["W2"]

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.array(x, dtype=np.float64)

        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, H, W, C), got {x.shape}")

        B, H, W, C = x.shape

        if H != self.image_size or W != self.image_size:
            raise ValueError(f"Expected H=W={self.image_size}, got H={H}, W={W}")
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if C != self.in_channels:
            raise ValueError(f"Expected C={self.in_channels}, got C={C}")

        patches = self._patchify(x)                 # (B, N, P*P*C)
        z = patches @ self.W_patch                  # (B, N, D)

        cls = np.repeat(self.cls_token, B, axis=0)  # (B, 1, D)
        z = np.concatenate([cls, z], axis=1)        # (B, N+1, D)

        z = z + self.pos_embed                      # (B, N+1, D)

        for l in range(self.depth):
            z_prime = z + self._msa(self._layer_norm(z), self.encoder_weights[l])
            z = z_prime + self._mlp(self._layer_norm(z_prime), self.encoder_weights[l])

        cls_out = z[:, 0, :]                        # (B, D)
        cls_out = self._layer_norm(cls_out)
        logits = cls_out @ self.W_head              # (B, num_classes)

        return logits