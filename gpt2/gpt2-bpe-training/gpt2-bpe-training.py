import torch
from typing import Tuple, List, Dict

def bpe_train(text: str, target_vocab_size: int) -> Tuple[List[Tuple[int, int]], Dict[int, bytes]]:
    """Returns: Tuple of (merge_rules, vocab) where merge_rules is a list of (id_a, id_b) tuples and vocab maps token IDs to bytes"""
    
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merge_rules: List[Tuple[int, int]] = []
    
    tokens = list(text.encode("utf-8"))
    
    if target_vocab_size <= 256:
        return merge_rules, vocab
    
    next_id = 256
    
    while len(vocab) < target_vocab_size:
        if len(tokens) < 2:
            break
        
        pair_counts = {}
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        
        if not pair_counts:
            break
        
        best_pair = min(pair_counts.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))[0]
        a, b = best_pair
        
        vocab[next_id] = vocab[a] + vocab[b]
        merge_rules.append((a, b))
        
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(next_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        
        tokens = new_tokens
        next_id += 1
    
    return merge_rules, vocab