import numpy as np


def softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    # implement stable softmax
    max = np.max(logits, keepdims=True) / temperature
    numerator = np.exp(logits / temperature - max)
    return numerator / np.sum(numerator)


def sample_next_token(
    logits: np.ndarray, temperature: float, top_k: int, top_p: float
) -> int:
    """
    logits: Shape (vocab_size,) representing the unnormalized prediction scores.
    Return: The index of the sampled token.
    """
    top_k = min(top_k, len(logits))

    probs = softmax(logits, temperature)

    sorted_indices = np.argsort(-probs, axis=0)

    top_k_mask = sorted_indices[:top_k]
    mask = np.zeros(logits.shape, dtype=np.bool)
    mask[top_k_mask] = True
    probs[~mask] = 0
    # this is making all indices not in top k to be 0

    # sorted probs
    sorted_probs = probs[sorted_indices]
    cumsum = np.cumsum(sorted_probs)

    # mask
    # probs  = [0.4, 0.3, 0.291, ] top p = 0.9
    # cumsum = [0.4, 0.7, 0.91, 1] so I need to choose all 3
    # find all > top p
    mask = cumsum > top_p  # these are the elementsj
    count = mask.sum()
    to_choose = len(sorted_indices) - count + 1
    indices_to_consider = sorted_indices[:to_choose]

    final_probs = probs[indices_to_consider] / np.sum(probs[indices_to_consider])

    return np.random.choice(indices_to_consider, p=final_probs)
