import numpy as np
from ml_implemented.decision_tree import DecisionTree
from sklearn import datasets
from sklearn.model_selection import train_test_split


class RandomForest:
    def __init__(
        self, n_trees: int = 10, min_samples_to_split: int = 10, max_depth: int = 10
    ):
        self.n_trees = n_trees
        self.min_samples_to_split = min_samples_to_split
        self.max_depth = max_depth
        self.trees = [
            DecisionTree(min_samples_to_split=min_samples_to_split, max_depth=max_depth)
            for _ in range(self.n_trees)
        ]

    def fit(self, X: np.ndarray, y: np.ndarray):
        for tree in self.trees:
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_sampled, y_sampled = X[indices], y[indices]
            tree.fit(X_sampled, y_sampled)

    def predict(self, X: np.ndarray) -> list:
        predictions_per_tree = []
        for tree in self.trees:
            predictions_per_tree.append(tree.predict(X))
        predictions_per_tree = np.swapaxes(
            np.array(predictions_per_tree), 0, 1
        )  # n, 10

        predictions = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x)), axis=1, arr=predictions_per_tree
        )
        return predictions


if __name__ == "__main__":
    X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    classifier = RandomForest()
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    accuracy = np.sum((np.array(predictions, dtype=int) == y_test).astype(int)) / len(
        y_test
    )

    print(f"accuracy is {accuracy}")
