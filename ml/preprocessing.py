import numpy as np
from itertools import combinations_with_replacement


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X = np.array(X)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1.0

        return self

    def transform(self, X):
        if self.mean_ is None or self.std_ is None:
            raise ValueError('scaler не обучен')
        X = np.array(X)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class PolynomialFeatures:

    def __init__(self, degree=2):
        self.degree = degree

    def transform(self, X):
        X = np.array(X)
        n_samples, n_features = X.shape
        poly_features = []

        for d in range(1, self.degree + 1):
            for comb in combinations_with_replacement(range(n_features), d):
                new_col = np.prod(X[:, comb], axis=1, keepdims=True)
                poly_features.append(new_col)
                
        return np.hstack(poly_features)