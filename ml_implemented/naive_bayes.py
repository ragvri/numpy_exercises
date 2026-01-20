import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import math


# p(Y|x) = P(x1|y) x P(x2|y) .... x P(X_n|Y) * P(Y)
class NaiveBayes:
    def __init__(self, n_class: int, n_features: int):
        self.n_features = n_features
        self.n_classes = n_class

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.priors = np.log(np.bincount(y) / len(y))

        means = []
        stds = []
        for label in range(self.n_classes):
            ids = np.where(y == label)[0]
            mean = np.mean(X[ids, :], axis=0)  # shape is n_features
            std = np.std(X[ids, :], axis=0)
            means.append(mean)
            stds.append(std)

        self.means = np.array(means)  # n_classes, features
        self.std = np.array(stds)

    def _pdf(self, mean: np.ndarray, std: np.ndarray, x: np.ndarray):
        numerator = np.exp(-((x - mean) ** 2) / (2 * std**2))
        denom = std * np.sqrt(2 * math.pi)
        return numerator / denom

    def predict(self, X: np.ndarray):
        probs = []
        for label in range(self.n_classes):
            print(label)
            prior = self.priors[label]
            posterior = np.sum(
                np.log(self._pdf(self.means[label], self.std[label], X)), axis=1
            )
            prob = prior + posterior
            probs.append(prob)
        probs = np.array(probs)  # class x bsz

        probs = np.swapaxes(probs, 0, 1)

        classes = np.argmax(probs, axis=1)
        return classes


if __name__ == "__main__":
    X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    for _ in range(10):
        classifier = NaiveBayes(n_features=10, n_class=2)
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)

        accuracy = np.sum((np.array(predictions, dtype=int) == y_test).astype(int)) / len(
            y_test
        )

        print(f"accuracy is {accuracy}")
