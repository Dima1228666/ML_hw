import numpy as np

from .base import BaseEstimator, ClassifierMixin, RegressorMixin


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
        X1 = np.asarray(X1)
        X2 = np.asarray(X2)

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


class SVC_binary(SVM, BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vectors_coef_ = None
        self.intercept_ = None
        self.support_vectors_ = None
        self.X_train = None
        self.coef_ = None

    def fit(self, X, y):
        self.X_train_ = X
        n_samples, _ = X.shape
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
        X = np.asarray(X)
        K = self.compute_kernel(X, self.X_train_)

        return K @ self.coef_ + self.intercept_

    def predict(self, X):
        X = np.asarray(X)

        return np.sign(self.decision_function(X))

class SVC(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            decision_function_shape: str='ovr',
              **kwargs):
        self.decision_function_shape = decision_function_shape
        self.kwargs = kwargs
        self.classes_ = None
        self.models_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
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
            

class SVR(SVM, BaseEstimator, RegressorMixin):
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
        n_samples, _ = X.shape
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
        X = np.asarray(X)
        K = self.compute_kernel(X, self.X_train_)

        return K @ self.coef_ + self.intercept_
