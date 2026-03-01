import numpy as np


class LinearNN:
    def __init__(self, d_in: int, d_out: int):
        """
        Single linear layer with bias absorbed into W,
        followed by a sigmoid activation.

        W shape: (d_in + 1, d_out)
        """
        self.W = np.random.randn(d_in + 1, d_out) * np.sqrt(2.0 / d_in)

        # Cache
        self.X_aug = None  # Augmented input [X | 1]
        self.Z = None  # Pre-activation: X_aug @ W
        self.A = None  # Post-activation: sigmoid(Z)

        # Gradient
        self.dW = None

    def _augment(self, X: np.ndarray) -> np.ndarray:
        ones = np.ones((X.shape[0], 1))
        return np.hstack([X, ones])

    def _sigmoid(self, Z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-Z))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward: A = sigmoid([X | 1] @ W)
        """
        self.X_aug = self._augment(X)  # (batch, d_in + 1)
        self.Z = self.X_aug @ self.W  # (batch, d_out)
        self.A = self._sigmoid(self.Z)  # (batch, d_out)
        return self.A

    def backward(self, dA: np.ndarray) -> np.ndarray:
        """
        Backward pass through activation + linear.

        Args:
            dA: Gradient w.r.t. this layer's output A
        Returns:
            dX: Gradient w.r.t. this layer's original input X
        """
        batch_size = self.X_aug.shape[0]

        # Sigmoid derivative: σ'(z) = σ(z)(1 - σ(z))
        dZ = dA * self.A * (1 - self.A)  # (batch, d_out)

        # Weight gradient
        self.dW = self.X_aug.T @ dZ  # (d_in + 1, d_out)

        # Input gradient (strip bias column)
        dX = (dZ @ self.W.T)[:, :-1]  # (batch, d_in)

        return dX

    def update(self, lr: float = 0.01):
        self.W -= lr * self.dW


def mse_loss(Y_pred: np.ndarray, Y_true: np.ndarray):
    """
    Mean Squared Error loss + its gradient w.r.t. Y_pred.

    Returns:
        loss:  scalar
        dA:    gradient of shape (batch, d_out) — kicks off backprop
    """
    batch_size = Y_true.shape[0]
    diff = Y_pred - Y_true
    loss = np.mean(diff**2)
    dA = (2 * diff) / batch_size  # ← this is the dY we were missing!
    return loss, dA


# Network: 3 → 4 → 1
layer1 = LinearNN(d_in=3, d_out=4)
layer2 = LinearNN(d_in=4, d_out=1)

# Dummy data
X = np.random.randn(32, 3)
Y = np.random.randn(32, 1)

# Training loop
for epoch in range(1000):
    # Forward
    A1 = layer1.forward(X)
    A2 = layer2.forward(A1)

    # Loss (this gives us the starting gradient!)
    loss, dA2 = mse_loss(A2, Y)

    # Backward
    dA1 = layer2.backward(dA2)
    _ = layer1.backward(dA1)

    # Update
    layer1.update(lr=0.01)
    layer2.update(lr=0.01)

    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")
