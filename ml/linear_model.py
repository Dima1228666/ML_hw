import numpy as np
from itertools import combinations_with_replacement
from .base import BaseEstimator, RegressorMixin, ClassifierMixin
from ml.metrics import r2_score
from ml.preprocessing import StandardScaler


class LinearRegression(BaseEstimator, RegressorMixin):

    def __init__(
        self,
        lr: float = 0.01,
        n_iters: int = 1000,
        fit_intercept: bool = True,
        regularization: str = None,
        alpha: float = 0.1
        ):

    
        self.lr = lr
        self.n_iters = n_iters
        self.fit_intercept = fit_intercept
        self.regularization = regularization
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        m = len(y)

        if self.fit_intercept:
            ones = np.ones((m, 1))
            X = np.c_[ones, X]
        else:
            X = X
        w = np.zeros(X.shape[1])

        for _ in range(self.n_iters):
            y_pred = X @ w
            gradient = (1 / m) * (X.T @ (y_pred - y))

            if self.regularization == 'L1':
                penalty = self.alpha * np.sign(w)
                if self.fit_intercept:
                    penalty[0] = 0
                gradient += penalty

            elif self.regularization == 'L2':
                penalty = self.alpha * w
                if self.fit_intercept:
                    penalty[0] = 0
                gradient += penalty

            elif self.regularization is None:
                pass

            else:
                raise ValueError('неправильный метод регуляризации')

            w = w - self.lr * gradient

        if self.fit_intercept:
            self.intercept_ = w[0]
            self.coef_ = w[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = w

        return self

    def predict(self, X):
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError('модель не обучена')
        X = np.array(X)

        return X @ self.coef_ + self.intercept_


class PolynomialRegression(BaseEstimator, RegressorMixin):

    def __init__(
        self,
        lr: float = 0.01,
        n_iters: int = 1000,
        degree: int = -1,
        fit_intercept: bool = True,
        regularization: str = None,
        alpha: float = 0.1
    ):
        self.lr = lr
        self.n_iters = n_iters
        self.fit_intercept = fit_intercept
        self.degree = degree
        self.regularization = regularization
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None
        self.scaler = None
        self.linear_model = LinearRegression(
            lr=lr, n_iters=n_iters, fit_intercept=fit_intercept,regularization=regularization,alpha=alpha)

    def create_polynomial_features(self, X):
        X = np.array(X)
        n_samples, n_features = X.shape
        poly_features = []

        for d in range(1, self.degree + 1):
            combinations = combinations_with_replacement(range(n_features), d)

            for comb in combinations:
                new_col = np.ones((n_samples, 1))
                for idx in comb:
                    new_col = new_col * X[:, idx : idx + 1]

                poly_features.append(new_col)

        return np.hstack(poly_features)

    def fit(self, X, y):
            X = np.array(X)
            y = np.array(y)
            scaler = StandardScaler()
            
            if self.degree == -1:
                idx = int(len(X) * 0.8)
                X_tr, X_test = X[:idx], X[idx:]
                y_tr, y_test = y[:idx], y[idx:]
                r2_list = []
                degrees = list(range(1, 8))

                for d in degrees:
                    self.degree = d
                    X_train_poly = self.create_polynomial_features(X_tr)
                    X_test_poly = self.create_polynomial_features(X_test)
                    X_train_poly = scaler.fit_transform(X_train_poly)
                    self.scaler = scaler
                    X_test_poly = scaler.transform(X_test_poly)
                    model = LinearRegression(
                        lr=self.lr, n_iters=self.n_iters, fit_intercept=self.fit_intercept,
                        regularization=self.regularization, alpha=self.alpha)
                    
                    model.fit(X_train_poly, y_tr)
                    r2 = r2_score(y_test, model.predict(X_test_poly))
                    r2_list.append(r2)

                self.degree = degrees[np.argmax(r2_list)]

            X_poly = self.create_polynomial_features(X)
            X_poly = scaler.fit_transform(X_poly)
            final_model = model = LinearRegression(
                                lr=self.lr, n_iters=self.n_iters, fit_intercept=self.fit_intercept,
                                regularization=self.regularization, alpha=self.alpha)
            final_model.fit(X_poly, y)
            self.coef_ = final_model.coef_
            self.intercept_ = final_model.intercept_

            return self

    def predict(self, X):
        if self.intercept_ is None or self.coef_ is None:
            raise ValueError('модель не обучена')
        
        X_poly = self.create_polynomial_features(X)
        X_poly = self.scaler.fit_transform(X_poly)

        return X_poly @ self.coef_ + self.intercept_


class LogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        lr: float = 0.01,
        n_iters: int = 1000,
        fit_intercept: bool = True,
        ):
        self.lr = lr
        self.n_iters = n_iters
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        m = len(y)

        if self.fit_intercept:
            ones = np.ones((m, 1))
            X = np.c_[ones, X]
        else:
            X = X
        w = np.zeros(X.shape[1])

        for _ in range(self.n_iters):
            y_pred = 1 / (1 + np.exp(-(X @ w)))

            gradient = (1 / m) * (X.T @ (y_pred - y))
            w = w - self.lr * gradient

        if self.fit_intercept:
            self.intercept_ = w[0]
            self.coef_ = w[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = w

        return self

    def predict(self, X):
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError('модель не обучена')

        predict = 1 / (1 + np.exp(-(X @ self.coef_ + self.intercept_)))
        predict = np.array(predict > 0.5)

        return predict