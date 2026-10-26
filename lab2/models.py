import numpy as np
from itertools import combinations_with_replacement


class LinearRegression:

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


class PolynomialRegression:

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
        scaler = StandardScaler()
        X_poly = scaler.fit_transform(X_poly)

        return X_poly @ self.coef_ + self.intercept_


class LogisticRegression:
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


class SVM:
    def __init__(
        self,
        C: float = 1.0,
        kernel: str = 'rbf',
        degree: int = 3,
        gamma: float = 0.1,
        coef0: float = 0.0,
        lr: float = 0.001,
        n_iters: int = 1000,
            ):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.lr = lr
        self.n_iters = n_iters

    def compute_kernel(self, X1, X2):
        X1 = np.array(X1)
        X2 = np.array(X2)

        if self.kernel == 'rbf':
            X1_norm = np.sum(X1**2, axis=1, keepdims=True)
            X2_norm = np.sum(X2**2, axis=1, keepdims=True).T
            product = X1 @ X2.T

            return np.exp(-(self.gamma * (X1_norm + X2_norm - 2 * product)))

        elif self.kernel == 'poly':
            return (X1 @ X2.T + self.coef0)**(self.degree)

        elif self.kernel == 'linear':
            return X1 @ X2.T
        
        else:
            raise ValueError('неправильное ядро')


class SVC_binary(SVM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vectors_coef_ = None
        self.intercept_ = None
        self.support_vectors_ = None
        self.X_train = None
        self.coef_ = None

    def fit(self, X, y):
        self.X_train_ = X
        n_samples, n_features = X.shape
        Gram = self.compute_kernel(X, X)
        w = np.zeros(n_samples)
        b = 0

        for _ in range(self.n_iters):
            margin = y * (Gram @ w + b)
            y_margin = y * (margin < 1.0)

            grad_w = w - (self.C / n_samples) * (Gram @ y_margin)
            grad_b = -(self.C / n_samples) * np.sum(y_margin)
            w = w - self.lr * grad_w
            b = b - self.lr * grad_b

        mask = y * (Gram @ w + b) < 1
        self.support_vectors_ = X[mask]
        self.intercept_ = b
        self.vectors_coef_ = w[mask]
        self.coef_ = w

        return self

    def decision_function(self, X):
        X = np.array(X)
        K = self.compute_kernel(X, self.X_train_)

        return K @ self.coef_ + self.intercept_

    def predict(self, X):
        X = np.array(X)

        return np.sign(self.decision_function(X))

class SVC:
    def __init__(
            self,
            decision_function_shape: str='ovr',
              **kwargs):
        self.decision_function_shape = decision_function_shape
        self.kwargs = kwargs
        self.classes_ = None
        self.models_ = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.classes_ = np.unique(y)
        k = len(np.unique(y))

        if self.decision_function_shape == 'ovr':
            self.models_ = []

            for cls in self.classes_:
                y_bin = np.where(y == cls, 1, -1)
                model = SVC_binary(**self.kwargs).fit(X, y_bin)
                self.models_.append(model)

        elif self.decision_function_shape == 'ovo':
            self.models_ = {}
            for i in range(k):
                for j in range(i + 1, k):
                    cls1, cls2 = self.classes_[i], self.classes_[j]
                    mask = (y == cls1) | (y == cls2)
                    X_bin = X[mask]
                    y_bin = np.where(y[mask] == cls1, 1, -1)
                    
                    model = SVC_binary(**self.kwargs).fit(X_bin, y_bin)
                    self.models_[(i, j)] = model
        else:
            raise ValueError('неверный метод')

        return self



    def predict(self, X):
        if self.decision_function_shape == 'ovr':
            scores = np.column_stack([model.decision_function(X) for model in self.models_])
            idx = np.argmax(scores, axis=1)

            return self.classes_[idx]
        
        if self.decision_function_shape == 'ovo':
            n_samples = X.shape[0]
            k = len(self.classes_)
            model_predict = np.zeros((n_samples, k))

            for (i, j), model in self.models_.items():
                predict = model.predict(X)
                model_predict[predict == 1, i] += 1
                model_predict[predict == -1, j] += 1

            best_idx = np.argmax(model_predict, axis=1)
            return self.classes_[best_idx]
            

class SVR(SVM):
    def __init__(self, epsilon: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.support_vectors_ = None
        self.vectors_coef_ = None
        self.X_train_ = None
        self.coef_ = None
        self.kwargs = kwargs

    def fit(self, X, y):
        self.X_train_ = X
        n_samples, n_features = X.shape
        Gram = self.compute_kernel(X, X)
        w = np.zeros(n_samples)
        b = 0

        for _ in range(self.n_iters):
            margin = Gram @ w + b - y
            mask_margin = np.zeros_like(margin)
            mask_margin[margin > self.epsilon] = 1.0
            mask_margin[margin < -self.epsilon] = -1.0

            grad_w = w + (self.C / n_samples) * (Gram @ mask_margin)
            grad_b = (self.C / n_samples) * np.sum(mask_margin)
            w = w - self.lr * grad_w
            b = b - self.lr * grad_b

        mask = np.abs(Gram @ w + b - y) >= self.epsilon
        self.support_vectors_ = X[mask]
        self.vectors_coef_ = w[mask]
        self.intercept_ = b
        self.coef_ = w

        return self

    def predict(self, X):
        X = np.array(X)
        K = self.compute_kernel(X, self.X_train_)

        return K @ self.coef_ + self.intercept_


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


def mean_absolute_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def mean_squared_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_avg = np.mean(y_true)
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_avg) ** 2)


def accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.sum(y_true == y_pred)/len(y_true)


def precision(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def recall(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)