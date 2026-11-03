import numpy as np

from .base import BaseEstimator, ClassifierMixin, RegressorMixin
from .tree import DecisionTreeClassifier, DecisionTreeRegressor


class RandomForestRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 5,
        max_features = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.estimators_ = []
        self.classes_ = None

    def _get_bootstrap(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.randint(0, n_samples, size=n_samples)
        return X[indices], y[indices]

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if self.max_features is None:
            self.max_features = max(1, X.shape[1] // 3)

        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features
            )
            b_X, b_y = self._get_bootstrap(X, y)
            tree.fit(b_X, b_y)
            self.estimators_.append(tree)

        return self

    def predict(self, X):
        X = np.asarray(X)
        tree_preds = np.array([tree.predict(X) for tree in self.estimators_])

        return np.mean(tree_preds, axis=0)


class RandomForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.estimators_ = []

    def _get_bootstrap(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.randint(0, n_samples, size=n_samples)
        return X[indices], y[indices]

    def fit(self, X, y):
        self.classes = np.unique(y)
        X = np.asarray(X)
        y = np.asarray(y)

        if self.max_features is None:
            self.max_features = max(1, int(np.sqrt(X.shape[1])))

        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features
            )
            b_X, b_y = self._get_bootstrap(X, y)
            tree.fit(b_X, b_y)
            self.estimators_.append(tree)

        return self

    def predict(self, X):
        X = np.asarray(X)
        preds = np.array([tree.predict(X) for tree in self.estimators_])
        final_pred = []

        for i in range(X.shape[0]):
            votes = preds[:, i]
            values, counts = np.unique(votes, return_counts=True)
            final_pred.append(values[np.argmax(counts)])

        return np.array(final_pred)