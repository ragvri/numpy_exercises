import numpy as np
import random


def softmax(logits: np.ndarray) -> np.ndarray:
    # n, c where n is the nubmer of points and c is the number of classes

    # lets do stable softmax where we subtrac the max first

    numerator = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # n, c
    denominator = np.sum(numerator, axis=1)  # n

    return numerator / denominator[:, None]


def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    # targets is n
    # -ylog(y)
    # for every row we know the correct class. And we see the prob of that correct class

    probs = softmax(logits)  # n, c
    return np.mean(-np.log(probs[np.arange(len(targets)), targets]), axis=0)


def ans(input) -> int:
    # [(1,3), (10, 20)]
    [0, 3, 23]
    cumsum = [0]
    current = 0
    for start, end in input:
        current += end - start + 1
        cumsum.append(current)

    total = cumsum[-1]

    r = random.randint(1, total)

    # find the first point in cumsum that is > r

    for i, suma in enumerate(cumsum):
        if suma >= r:
            break

    total_to_move = r - cumsum[i - 1] - 1

    element = input[i - 1][0] + total_to_move

    return element


items = ["a", "b", "c"]
weights = [1, 2, 7]  # 'a' has 10% chance, 'b' has 20%, 'c' has 70%


def weighted_sample(items: list, weights: list) -> str:
    # You can only use randint(a, b) where a <= x < b
    cumsum = [0]
    total = 0
    for weight in weights:
        total += weight
        cumsum.append(weight)

    r = random.randint(1, cumsum[-1])

    for i, suma in enumerate(cumsum):
        if suma >= r:
            break
    element = items[i - 1]
    return element
