import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from torch import nn, optim


class LinearRegression:
    def __init__(self):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray):
        if X.ndim == 1:
            X = X[:, None]
        # flatten to 2d
        self.X_train = rearrange(X, "m ... -> m (...)")
        # self.X_train = X.reshape(X.shape[0], -1)

        # from n x d to n x (d+1)
        self.X_train = np.concat(
            [self.X_train, np.ones(shape=(self.X_train.shape[0], 1))], axis=1
        )

        # m x 1
        self.y_train = y[:, None]

        self.W = np.linalg.pinv(self.X_train.T @ self.X_train) @ (
            self.X_train.T @ self.y_train
        )
        assert self.W.shape == (self.X_train.shape[-1], 1)

    def predict(self, points: np.ndarray) -> np.ndarray:
        # points is n x d

        points = points.reshape(points.shape[0], -1)

        points = np.concat([points, np.ones(shape=(points.shape[0], 1))], axis=1)

        return points @ self.W


class LinearRegressionGradientDescent:
    def __init__(self, iterations: int = 10_000, lr=1e-3):
        self.iterations = iterations
        self.lr = lr
        self.W = None

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

        print(f"shapes: {X.shape}, {self.W.shape}, {y.shape}")
        for iteration in range(self.iterations):
            grad = 1 / X.shape[0] * (X.T) @ (X @ self.W - y)

            self.W = self.W - self.lr * grad

            prediction = X @ self.W

            mse = np.mean(np.sum((prediction - y) ** 2, axis=1), axis=0)

            if iteration % 500 == 0:
                print(f"mse:{mse} iteration:{iteration}")

    def predict(self, X: np.ndarray):
        # make X 2d
        if X.ndim == 1:
            X = X[:, None]
        else:
            X = X.reshape(X.shape[0], -1)

        # add ones to X for bias
        X = np.concat([X, np.ones(shape=(X.shape[0], 1))], axis=1)

        prediction = X @ self.W

        return prediction


class LinearRegressionPytorch(nn.Module):
    def __init__(self, iterations: int = 10_000, lr: float = 1e-3):
        super().__init__()
        self.iterations = iterations
        self.lr = lr

        self.loss = nn.MSELoss()

    def fit(self, X: Float[np.ndarray, "batch ..."], y: Float[np.ndarray, "batch ..."]):
        # convert X to 2d
        if X.ndim == 1:
            X = X[:, None]
        else:
            X = X.reshape(X.shape[0], -1)

        y = y.reshape(y.shape[0], -1)
        y = torch.from_numpy(y)
        y = y.to(torch.float)

        self.linear = nn.Linear(X.shape[1], 1, bias=True)
        self.optimizer = optim.SGD(lr=self.lr, params=self.linear.parameters())

        for iteration in range(self.iterations):
            self.optimizer.zero_grad()
            logits = self.forward(X)

            loss = self.loss(logits, y)

            loss.backward()
            self.optimizer.step()

            if iteration % 500 == 0:
                print(f"mse: {loss}, iteration: {iteration}")

    def forward(self, X: Float[np.ndarray, "batch ..."]):  # noqa: F722
        # convert numpy to tensor
        X = torch.from_numpy(X)  # ty:ignore[invalid-assignment]
        # X = X.to(torch.float)

        return self.linear(X)

    def predict(self, X: Float[np.ndarray, "batch ..."]):
        # convert X to 2d
        if X.ndim == 1:
            X = X[:, None]
        else:
            X = X.reshape(X.shape[0], -1)

        self.eval()
        with torch.no_grad():
            return self.forward(X).squeeze(axis=1).numpy()


if __name__ == "__main__":
    X_train = np.array(
        [[0, 0], [1, 1], [2, 2], [10, 10], [11, 11], [12, 12]], dtype=np.float32
    )
    y = np.array([0, 1, 2, 10, 11, 12], dtype=np.float32)

    for type, linear_regression in [
        # ("closed_form", LinearRegression()),
        # ("grad_descent", LinearRegressionGradientDescent()),
        ("pytorch", LinearRegressionPytorch()),
    ]:
        linear_regression.fit(X_train, y)

        point = np.array([[0.5, 0.5]], dtype=np.float32)

        prediction = linear_regression.predict(point)

        print(f"type: {type}. prediction: {prediction}")
