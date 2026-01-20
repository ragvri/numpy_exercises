import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, k: int = 5, max_iterations: int = 1000):
        self.k = k
        self.max_iterations = max_iterations
        self.centers = None

    def get_distance(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        # X is m, d
        # Y is n, d
        X = X[:, None, :]
        Y = Y[None, :, :]
        distance = np.sqrt(np.sum((X - Y) ** 2, axis=2))
        return distance

    def fit(self, X: np.ndarray):
        # choose k points as the initial centers
        self.centers = X[np.random.choice(len(X), self.k)]
        self.clusters = np.zeros(len(X))

        for _ in range(self.max_iterations):
            distance = self.get_distance(X, self.centers)  # n x k
            new_clusters = np.argmin(distance, axis=1)  # n

            if np.all(new_clusters == self.clusters):
                break

            self.clusters = new_clusters

            centers = []
            for i in range(self.k):
                mask = self.clusters == i
                if np.any(mask):
                    new_center = X[mask].mean(axis=0)
                else:
                    # randomly initalize to a new point
                    new_center = X[np.random.choice(len(X), size=1)]
                centers.append(new_center)

            self.centers = np.array(centers)  # k, d

    def predict(self, X: np.ndarray):
        assert self.centers is not None
        distance = self.get_distance(X, self.centers)  # n x k

        cluster = np.argmin(distance, axis=1)
        return cluster


if __name__ == "__main__":
    X, y = make_blobs(n_samples=100, n_features=2, centers=5)

    kmeans = KMeans()

    kmeans.fit(X)

    predictions = kmeans.predict(X)

    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis")
    plt.title("Ground Truth")

    plt.subplot(1, 3, 2)
    plt.scatter(X[:, 0], X[:, 1], c=predictions, cmap="viridis")
    plt.scatter(kmeans.centers[:, 0], kmeans.centers[:, 1], c="red", s=200)
    plt.title("KMeans Predictions")

    plt.show()
