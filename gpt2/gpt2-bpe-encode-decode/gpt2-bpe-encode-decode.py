import torch
from typing import Tuple, List, Dict

def bpe_encode(text: str, merge_rules: List[Tuple[int, int]], vocab: Dict[int, bytes]) -> List[int]:
    """
    Returns: List of integer token IDs
    """
    tokens = list(text.encode("utf-8"))

    bytes_to_id = {byte_seq: token_id for token_id, byte_seq in vocab.items()}

    for a, b in merge_rules:
        merged_bytes = vocab[a] + vocab[b]
        new_id = bytes_to_id[merged_bytes]

        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(new_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens

    return tokens

def bpe_decode(token_ids: List[int], vocab: Dict[int, bytes]) -> str:
    """
    Returns: Decoded UTF-8 string
    """
    byte_string = b"".join(vocab[token_id] for token_id in token_ids)
    return byte_string.decode("utf-8")