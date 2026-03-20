import numpy as np


class CollaborativeFiltering:
    """Barebones matrix factorization with NumPy.

    Ratings matrix shape: (num_users, num_items).
    Missing entries must be np.nan.
    """

    def __init__(self, n_factors: int = 10, learning_rate: float = 0.01, epochs: int = 200):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.user_factors = None
        self.item_factors = None

    def fit(self, ratings: np.ndarray) -> None:
        if ratings.ndim != 2:
            raise ValueError("ratings must be a 2D array")

        ratings = ratings.astype(float)
        n_users, n_items = ratings.shape
        observed = np.argwhere(~np.isnan(ratings))

        if observed.size == 0:
            raise ValueError("ratings has no observed values")

        self.user_factors = 0.1 * np.random.standard_normal((n_users, self.n_factors))
        self.item_factors = 0.1 * np.random.standard_normal((n_items, self.n_factors))

        for epoch in range(self.epochs):
            # observed is (num_observed, 2): [user_idx, item_idx] per known rating.
            user_idx = observed[:, 0]
            item_idx = observed[:, 1]

            # target: (num_observed,)
            target = ratings[user_idx, item_idx]
            # user_vecs/item_vecs: (num_observed, n_factors)
            user_vecs = self.user_factors[user_idx]
            item_vecs = self.item_factors[item_idx]

            # pred/error: (num_observed,)
            pred = np.sum(user_vecs * item_vecs, axis=1)
            error = target - pred

            # grad tensors: (num_observed, n_factors)
            user_grad = self.learning_rate * error[:, None] * item_vecs
            item_grad = self.learning_rate * error[:, None] * user_vecs

            # Scatter-add back into full factor tables:
            # user_factors: (n_users, n_factors), item_factors: (n_items, n_factors)
            np.add.at(self.user_factors, user_idx, user_grad)
            np.add.at(self.item_factors, item_idx, item_grad)

            pred_after = np.sum(
                self.user_factors[user_idx] * self.item_factors[item_idx], axis=1
            )
            error_after = target - pred_after
            mse = np.mean(error_after**2)
            print(f"epoch={epoch + 1}, mse={mse:.6f}")

    def predict(self, user_idx: int, item_idx: int) -> float:
        self._check_fitted()
        return float(np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))

    def predict_all(self) -> np.ndarray:
        self._check_fitted()
        return self.user_factors @ self.item_factors.T

    def _check_fitted(self) -> None:
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("model is not fitted. Call fit(...) first")


if __name__ == "__main__":
    ratings_matrix = np.array(
        [
            [5, 4, np.nan, 1],
            [4, np.nan, np.nan, 1],
            [1, 1, np.nan, 5],
            [np.nan, 1, 5, 4],
            [np.nan, np.nan, 4, 5],
        ],
        dtype=float,
    )

    model = CollaborativeFiltering(n_factors=3, learning_rate=0.01, epochs=300)
    model.fit(ratings_matrix)

    print("Predicted full matrix:")
    print(np.round(model.predict_all(), 2))
