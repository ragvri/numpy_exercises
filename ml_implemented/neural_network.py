import numpy as np


class LinearNN:
    def __init__(self, d_in: int, d_out: int, activation: str = "sigmoid"):
        """
        Single linear layer with bias absorbed into W,
        followed by an activation function.

        W shape: (d_in + 1, d_out)
        activation: "sigmoid", "relu", or "linear"
        """
        self.activation = activation
        self.W = np.random.randn(d_in + 1, d_out) * np.sqrt(2.0 / d_in)

        # Cache
        self.X_aug = None  # Augmented input [X | 1]
        self.Z = None  # Pre-activation: X_aug @ W
        self.A = None  # Post-activation

        # Gradient
        self.dW = None

    def _augment(self, X: np.ndarray) -> np.ndarray:
        ones = np.ones((X.shape[0], 1))
        return np.concatenate([X, ones], axis=1)

    def _activate(self, Z: np.ndarray) -> np.ndarray:
        if self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-Z))
        elif self.activation == "relu":
            return np.maximum(0, Z)
        else:  # linear
            return Z

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X_aug = self._augment(X)  # (batch, d_in + 1)
        self.Z = self.X_aug @ self.W  # (batch, d_out)
        self.A = self._activate(self.Z)
        return self.A

    def backward(self, dA: np.ndarray) -> np.ndarray:
        """
        Backward pass through activation + linear.

        Args:
            dA: Gradient w.r.t. this layer's output A
        Returns:
            dX: Gradient w.r.t. this layer's original input X
        """
        # Activation derivative
        if self.activation == "sigmoid":
            dZ = dA * self.A * (1 - self.A)
        elif self.activation == "relu":
            dZ = dA * (self.Z > 0).astype(float)
        else:  # linear
            dZ = dA

        # Weight gradient
        self.dW = self.X_aug.T @ dZ  # (d_in + 1, d_out)

        # Input gradient (strip bias column)
        dX = (dZ @ self.W.T)[:, :-1]  # (batch, d_in)

        return dX

    def update(self, lr: float = 0.01):
        self.W -= lr * self.dW


def mse_loss(Y_pred: np.ndarray, Y_true: np.ndarray):
    batch_size = Y_true.shape[0]
    diff = Y_pred - Y_true
    loss = np.mean(diff**2)
    dA = (2 * diff) / batch_size
    return loss, dA


def softmax_forward(logits: np.ndarray) -> np.ndarray:
    """
    Args:
        logits: (batch, num_classes) raw scores from final linear layer
    Returns:
        probs: (batch, num_classes) probability distribution per sample
    """
    # Subtract max for numerical stability (prevents exp overflow)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def softmax_backward(dProbs: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """
    Args:
        dProbs: (batch, num_classes) gradient w.r.t. softmax output
        probs: (batch, num_classes) cached softmax output from forward
    Returns:
        dLogits: (batch, num_classes) gradient w.r.t. logits

    dy_i/dx_j = y_i(δ_ij - y_j)
    Jacobian: J = diag(y) - y @ y^T
    dLogits = dProbs @ J  (per sample)

    Simplification: dProbs @ J expands to
      dL/dx_j = y_j * (dL/dy_j - sum_i(dL/dy_i * y_i))
             = probs * (dProbs - sum(dProbs * probs))
    which avoids building the (C, C) Jacobian — O(C) instead of O(C^2).
    """
    batch_size = probs.shape[0]
    # J = diag(y) - y @ y^T per sample
    # probs[:, :, None] is (batch, C, 1), probs[:, None, :] is (batch, 1, C)
    jacobian = np.eye(probs.shape[1])[None, :, :] * probs[:, :, None] - probs[:, :, None] @ probs[:, None, :]  # (batch, C, C)
    # dLogits = dProbs @ J -> (batch, 1, C) @ (batch, C, C) -> (batch, 1, C) -> squeeze to (batch, C)
    dLogits = (dProbs[:, None, :] @ jacobian).squeeze(1)
    return dLogits
    # Simplified O(C) version (equivalent, avoids building the Jacobian):
    # dot = np.sum(dProbs * probs, axis=1, keepdims=True)  # sum_i(dL/dy_i * y_i), shape (batch, 1)
    # return probs * (dProbs - dot)  # y_j * (dL/dy_j - dot), shape (batch, C)


def cross_entropy_forward(probs: np.ndarray, Y_true: np.ndarray):
    """
    Args:
        probs: (batch, num_classes) softmax output
        Y_true: (batch,) integer class labels
    Returns:
        loss: scalar average cross-entropy
    """
    batch_size = Y_true.shape[0]
    correct_probs = probs[np.arange(batch_size), Y_true]
    # in practice you'd clamp: np.log(np.clip(correct_probs, 1e-12, None))
    loss = -np.mean(np.log(correct_probs))
    return loss


def cross_entropy_backward(probs: np.ndarray, Y_true: np.ndarray) -> np.ndarray:
    """
    Args:
        probs: (batch, num_classes) softmax output
        Y_true: (batch,) integer class labels
    Returns:
        dProbs: (batch, num_classes) gradient w.r.t. probs

    dL/dp_i = 0 for non-true classes, -1/p_true for the true class
    averaged over batch
    """
    batch_size = Y_true.shape[0]
    dProbs = np.zeros_like(probs)
    dProbs[np.arange(batch_size), Y_true] = -1.0 / probs[np.arange(batch_size), Y_true]
    dProbs /= batch_size
    return dProbs


# ── Regression example: 3 → 4 → 1 ──
print("=== Regression (MSE + ReLU/Linear) ===")
layer1 = LinearNN(d_in=3, d_out=4, activation="relu")
layer2 = LinearNN(d_in=4, d_out=1, activation="linear")

X_reg = np.random.randn(32, 3)
Y_reg = np.random.randn(32, 1)

for epoch in range(1000):
    A1 = layer1.forward(X_reg)
    A2 = layer2.forward(A1)
    loss, dA2 = mse_loss(A2, Y_reg)
    dA1 = layer2.backward(dA2)
    _ = layer1.backward(dA1)
    layer1.update(lr=0.01)
    layer2.update(lr=0.01)
    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")


# ── Classification example: 4 → 8 → 3 (3 classes) ──
print("\n=== Classification (Softmax + Cross-Entropy) ===")
clf_layer1 = LinearNN(d_in=4, d_out=8, activation="relu")
clf_layer2 = LinearNN(d_in=8, d_out=3, activation="linear")  # logits, no activation

X_cls = np.random.randn(64, 4)
Y_cls = np.random.randint(0, 3, size=64)  # integer class labels

for epoch in range(1000):
    # Forward
    H = clf_layer1.forward(X_cls)
    logits = clf_layer2.forward(H)
    probs = softmax_forward(logits)

    # Loss
    loss = cross_entropy_forward(probs, Y_cls)

    # Backward — cross-entropy then softmax then layers
    dProbs = cross_entropy_backward(probs, Y_cls)
    dLogits = softmax_backward(dProbs, probs)
    dH = clf_layer2.backward(dLogits)
    _ = clf_layer1.backward(dH)

    # Update
    clf_layer1.update(lr=0.1)
    clf_layer2.update(lr=0.1)

    if epoch % 200 == 0:
        preds = np.argmax(probs, axis=1)
        acc = np.mean(preds == Y_cls)
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Acc: {acc:.2%}")
