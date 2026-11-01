import numpy as np
from .base import BaseEstimator, RegressorMixin, ClassifierMixin


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    @property
    def is_leaf(self):
        return self.value is not None


class DecisionTreeRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def _mse(self, y):
        if len(y) == 0:
            return 0.0
        return np.mean((y - np.mean(y)) ** 2)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        best_quality = -1.0
        best_feature = None
        best_threshold = None
        cur_mse = self._mse(y)

        for feature_idx in range(n_features):
            X_col = X[:, feature_idx]
            thresholds = np.unique(X_col)

            for threshold in thresholds:
                left_mask = X_col <= threshold
                right_mask = X_col > threshold

                if (np.sum(left_mask) < self.min_samples_leaf or 
                    np.sum(right_mask) < self.min_samples_leaf):
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]
                n_left, n_right = len(y_left), len(y_right)
                quality = cur_mse - (n_left / n_samples) * self._mse(y_left) - (n_right / n_samples) * self._mse(y_right)

                if quality > best_quality:
                    best_quality = quality
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_quality

    def _build_tree(self, X, y, depth=0):
        n_samples = X.shape[0]

        if (self.max_depth is not None and depth >= self.max_depth) or \
           (n_samples < self.min_samples_split) or \
           (len(np.unique(y)) == 1):
            return Node(value=np.mean(y))

        best_feature, best_threshold, best_quality = self._best_split(X, y)

        if best_quality <= 0 or best_feature is None:
            return Node(value=np.mean(y))
        
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = X[:, best_feature] > best_threshold
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature=best_feature, threshold=best_threshold, left=left_tree, right=right_tree)

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.root = self._build_tree(X, y)

        return self

    def _predict_sample(self, node, x):
        if node.is_leaf:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._predict_sample(node.left, x)
        else:
            return self._predict_sample(node.right, x)

    def predict(self, X):
        X = np.asarray(X)

        return np.array([self._predict_sample(self.root, x) for x in X])


class DecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        criterion='gini',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        self.classes_ = None

    def _calculate_impurity(self, y):
        if len(y) == 0:
            return 0.0

        values, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)

        if self.criterion == 'gini':
            return 1.0 - np.sum(probabilities ** 2)
        
        elif self.criterion == 'entropy':
            if probabilities == 0:
                return 0
            return -np.sum(probabilities * np.log2(probabilities))
        
        else:
            raise ValueError(f'неправильный критерий')

    def _most_common_class(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        best_quality = -1.0
        best_feature = None
        best_threshold = None
        cur_impurity = self._calculate_impurity(y)

        for feature_idx in range(n_features):
            X_col = X[:, feature_idx]
            thresholds = np.unique(X_col)

            for threshold in thresholds:
                left_mask = X_col <= threshold
                right_mask = X_col > threshold

                if (np.sum(left_mask) < self.min_samples_leaf or 
                    np.sum(right_mask) < self.min_samples_leaf):
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]
                n_left, n_right = len(y_left), len(y_right)
                quality = (
                    cur_impurity 
                    - (n_left / n_samples) * self._calculate_impurity(y_left)
                    - (n_right / n_samples) * self._calculate_impurity(y_right))

                if quality > best_quality:
                    best_quality = quality
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_quality

    def _build_tree(self, X, y, depth=0):
        n_samples = X.shape[0]

        if (self.max_depth is not None and depth >= self.max_depth) or \
           (n_samples < self.min_samples_split) or \
           (len(np.unique(y)) == 1):
            return Node(value=self._most_common_class(y))
        
        best_feature, best_threshold, best_quality = self._best_split(X, y)

        if best_quality <= 0 or best_feature is None:
            return Node(value=self._most_common_class(y))

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = X[:, best_feature] > best_threshold
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature=best_feature, threshold=best_threshold, left=left_tree, right=right_tree)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.root = self._build_tree(X, y)
        return self

    def _predict_sample(self, node, x):
        if node.is_leaf:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._predict_sample(node.left, x)
        else:
            return self._predict_sample(node.right, x)

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict_sample(self.root, x) for x in X])