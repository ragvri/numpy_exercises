import numpy as np
import torch
from jaxtyping import Float
from torch import nn, optim


class LogisticRegression:
    def __init__(self, iterations: int = 10_000, lr=1e-3):
        self.iterations = iterations
        self.lr = lr
        self.W = None

    def sigmoid(self, logits):
        probs = 1 / (
            1
            + np.exp(
                -1 * logits,
            )
        )
        return probs

    def fit(self, X: np.ndarray, y: np.ndarray):
        # make X 2d
        if X.ndim == 1:
            X = X[:, None]
        else:
            X = X.reshape(X.shape[0], -1)

        # add ones to X for bias
        X = np.concat([X, np.ones(shape=(X.shape[0], 1))], axis=1)

        y = y.reshape(y.shape[0], -1)  # n x 1
        self.W = np.random.normal(size=(X.shape[1], 1))

        for iteration in range(self.iterations):
            logits = X @ self.W
            probs = self.sigmoid(logits)

            grad = 1 / X.shape[0] * (X.T) @ (probs - y)

            self.W = self.W - self.lr * grad

            cross_entropy = (
                1
                / X.shape[0]
                * np.sum(y * np.log(probs) + (1 - y) * (np.log(1 - probs)), axis=0)
            ) * -1

            if iteration % 500 == 0:
                print(f"BCE:{cross_entropy} iteration:{iteration}")

    def predict(self, X: np.ndarray) -> list:
        # make X 2d
        if X.ndim == 1:
            X = X[:, None]
        else:
            X = X.reshape(X.shape[0], -1)

        # add ones to X for bias
        X = np.concat([X, np.ones(shape=(X.shape[0], 1))], axis=1)

        logits = X @ self.W
        prob = self.sigmoid(logits)

        classes = (prob > 0.5).astype(int).squeeze(axis=1).tolist()

        return classes


class LogisticRegressionPytorch(nn.Module):
    def __init__(self, iterations: int = 10_000, lr: float = 1e-3):
        super().__init__()
        self.iterations = iterations
        self.lr = lr

        self.loss = nn.BCELoss()

    def fit(self, X: Float[np.ndarray, "batch ..."], y: Float[np.ndarray, "batch ..."]):
        # convert X to 2d
        if X.ndim == 1:
            X = X[:, None]
        else:
            X = X.reshape(X.shape[0], -1)

        y = y.reshape(y.shape[0], -1)
        y = torch.from_numpy(y)
        y = y.to(torch.float32)

        self.linear = nn.Linear(X.shape[1], 1, bias=True)
        self.optimizer = optim.SGD(lr=self.lr, params=self.linear.parameters())

        for iteration in range(self.iterations):
            self.optimizer.zero_grad()
            logits = self.forward(X)

            loss = self.loss(torch.sigmoid(logits), y)

            loss.backward()
            self.optimizer.step()

            if iteration % 500 == 0:
                print(f"bce: {loss}, iteration: {iteration}")

    def forward(self, X: Float[np.ndarray, "batch ..."]):  # noqa: F722
        # convert numpy to tensor
        X = torch.from_numpy(X)  # ty:ignore[invalid-assignment]

        return self.linear(X)

    def predict(self, X: Float[np.ndarray, "batch ..."]):
        # convert X to 2d
        if X.ndim == 1:
            X = X[:, None]
        else:
            X = X.reshape(X.shape[0], -1)

        self.eval()
        with torch.no_grad():
            logits = self.forward(X)
            probs = (torch.sigmoid(logits) > 0.5).int().squeeze(axis=1).tolist()
            return probs


if __name__ == "__main__":
    X_train = np.array(
        [[0, 0], [1, 1], [2, 2], [10, 10], [11, 11], [12, 12]], dtype=np.float32
    )
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)

    for type, logistic_regression in [
        ("numpy", LogisticRegression()),
        ("torch", LogisticRegressionPytorch()),
    ]:
        logistic_regression.fit(X_train, y)

        point = np.array([[0.5, 0.5], [92, 80]], dtype=np.float32)

        prediction = logistic_regression.predict(point)

        print(f"type: {type}. prediction: {prediction}")
