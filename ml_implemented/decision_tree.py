import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


class Node:
    def __init__(self, threshold=None, feature_to_split=None, val=None):
        self.threshold = threshold
        self.feature_to_split = feature_to_split
        self.left_node = None
        self.right_node = None

        self.val = val


class DecisionTree:
    def __init__(self, min_samples_to_split: int = 10, max_depth: int = 10):
        self.min_samples_to_split = min_samples_to_split
        self.max_depth = max_depth
        self.root = None

    def _get_entropy(self, y: np.ndarray) -> float:
        assert y.ndim == 1
        counts = np.bincount(y)
        non_zero_ids = np.where(counts > 0)

        probs = counts[non_zero_ids] / len(y)  # c where c is the number of classes

        entropy = -np.sum(probs * np.log(probs), axis=0)

        return entropy

    def get_information_gain(self, X, y: np.ndarray, f_id, threshold) -> float:
        parent_entropy = self._get_entropy(y)

        left_child_ids = np.where(X[:, f_id] < threshold)[0]
        right_child_ids = np.where(X[:, f_id] >= threshold)[0]

        if len(left_child_ids) == 0 or len(right_child_ids) == 0:
            return 0

        left_child_entropy = self._get_entropy(y[left_child_ids]) * (
            len(left_child_ids) / len(y)
        )
        right_child_entropy = (
            self._get_entropy(y[right_child_ids]) * len(right_child_ids) / len(y)
        )

        return parent_entropy - (left_child_entropy + right_child_entropy)

    def _get_best_split(self, X, y):
        best_gain = -1
        best_feature = best_threshold = None

        for f_id in range(X.shape[1]):
            thresholds = np.unique(X[:, f_id])
            for threshold in thresholds:
                gain = self.get_information_gain(X, y, f_id, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = f_id
                    best_threshold = threshold

        return best_feature, best_threshold

    def get_most_common(self, y: np.ndarray) -> int:
        return int(np.argmax(np.bincount(y)))

    def _grow_tree(self, X, y: np.ndarray, depth=0):
        # for every feature
        # go to every threshold of that feature
        # find the IG with that feature
        # split on the best threshold, best_feature
        # everything smaller goes to the left
        # everything bigger goes to the right
        if (
            depth == self.max_depth
            or len(np.unique(y)) == 1
            or X.shape[0] <= self.min_samples_to_split
        ):
            label = self.get_most_common(y)
            node = Node(val=label)
            return node
        best_feature, best_threshold = self._get_best_split(X, y)

        left_child_ids = np.where(X[:, best_feature] < best_threshold)[0]
        right_child_ids = np.where(X[:, best_feature] >= best_threshold)[0]

        node = Node(threshold=best_threshold, feature_to_split=best_feature)

        node.left_node = self._grow_tree(
            X[left_child_ids], y[left_child_ids], depth + 1
        )

        node.right_node = self._grow_tree(
            X[right_child_ids], y[right_child_ids], depth + 1
        )
        return node

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.root = self._grow_tree(X, y)

    def _traverse(self, node: Node | None, X: np.ndarray):
        if node.val is not None:
            return node
        best_feature, best_threshold = node.feature_to_split, node.threshold

        if X[best_feature] < best_threshold:
            return self._traverse(node.left_node, X)

        return self._traverse(node.right_node, X)

    def predict(self, X: np.ndarray) -> list:
        predictions = []
        for x in X:
            leaf = self._traverse(self.root, x)
            predictions.append(leaf.val)

        return predictions


if __name__ == "__main__":
    X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    classifier = DecisionTree()
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    accuracy = np.sum((np.array(predictions, dtype=int) == y_test).astype(int)) / len(
        y_test
    )

    print(f"accuracy is {accuracy}")
