import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    positions = np.arange(seq_length)[:, np.newaxis]          # (seq_length, 1)
    div_terms = np.exp(
        np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
    )                                                         # (d_model/2,)

    pe = np.zeros((seq_length, d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(positions * div_terms)
    pe[:, 1::2] = np.cos(positions * div_terms)

    return pe
