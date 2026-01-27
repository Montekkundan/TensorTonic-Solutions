import numpy as np

def compute_advantage(states, rewards, V, gamma):
    """
    Returns: A (NumPy array of advantages)
    """
    T = len(states)
    advantages = np.empty(T, dtype=float)

    G = 0.0
    for t in range(T - 1, -1, -1):
        G = rewards[t] + gamma * G
        advantages[t] = G - V[states[t]]

    return advantages