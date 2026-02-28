import numpy as np
from collections import defaultdict
from einops import rearrange


class KNN:
    def __init__(self, k: int = 3):
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray):
        if X.ndim == 1:
            X = X[:, None]
        # flatten to 2d
        self.X_train = rearrange(X, "m ... -> m (...)")
        # self.X_train = X.reshape(X.shape[0], -1)

        self.y_train = y

    def get_distance(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        # X is n,m
        # Y is k, m
        # output needs to be n,k
        X = X[:, None, :]  # n,1, m
        Y = Y[None, :, :]  # 1, k, m

        # now broadcastable

        distance = np.sqrt(np.sum((X - Y) ** 2, axis=2))
        return distance

    def predict(self, points: np.ndarray) -> list:
        # for each point in X, find distance to all points in X_train
        # then find the closest K points in X_train
        # use the y labels from them and find the most freq one as the
        # prediction for this x

        distances = self.get_distance(X=points, Y=self.X_train)

        # m,k
        indices = np.argpartition(distances, axis=-1, kth=self.k)[:, : self.k]

        neighbor_labels = self.y_train[indices]  # (n, k) -> using advanced indexing with broadcast

        num_classes = int(self.y_train.max()) + 1
        predictions = [
            np.argmax(np.bincount(row, minlength=num_classes))
            for row in neighbor_labels
        ]
        return predictions

if __name__ == "__main__":
    X_train = np.array(
        [[0, 0], [1, 1], [2, 2], [10, 10], [11, 11], [12, 12]], dtype=int
    )
    y = np.array([0, 0, 0, 1, 1, 1])

    knn = KNN(k=1)
    knn.fit(X_train, y)

    point = np.array([[0.5, 0.5]])

    prediction = knn.predict(point)

    assert prediction[0] == 0
