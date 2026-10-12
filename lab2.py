import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import sklearn.metrics as sk_metrics


class CustomLinearRegression:

    def __init__(
        self,
        lr: float = 0.01,
        n_iter: int = 1000,
        fit_intercept: bool = True,
    ):
        self.lr = lr
        self.n_iter = n_iter
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

        for _ in range(self.n_iter):
            y_pred = X @ w

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

        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return X @ self.coef_ + self.intercept_


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



if __name__ == '__main__':
    X = np.random.uniform(-5, 5, size=(100, 1))
    noise = np.random.normal(0, 2, size=(100, 1))
    y = (5 + 2 * X + noise).ravel()

    my_model = CustomLinearRegression(lr=0.1, n_iter=1000)
    my_model.fit(X, y)
    my_pred = my_model.predict(X)

    sk_model = LinearRegression()
    sk_model.fit(X, y)
    sk_pred = sk_model.predict(X)

    print('коэффициенты моделей:')
    print(f'intercept')
    print(f'наша модель: {my_model.intercept_:.4f}')
    print(f'scikit-learn: {sk_model.intercept_:.4f}')

    print(f'\ncoef')
    print(f'наша модель: {my_model.coef_[0]:.4f}')
    print(f'scikit-learn: {sk_model.coef_[0]:.4f}')
    print('=' * 50)

    print('сравнение метрик:')
    print(f'MAE:')
    print(f'наша функция: {mean_absolute_error(y, my_pred):.4f}')
    print(f'scikit-learn: {sk_metrics.mean_absolute_error(y, my_pred):.4f}')

    print(f'\nMSE:')
    print(f'наша функция: {mean_squared_error(y, my_pred):.4f}')
    print(f'scikit-learn: {sk_metrics.mean_squared_error(y, my_pred):.4f}')

    print(f'\nR2-score:')
    print(f'наша функция: {r2_score(y, my_pred):.4f}')
    print(f'scikit-learn: {sk_metrics.r2_score(y, my_pred):.4f}')


    plt.figure(figsize=(10, 6))

    plt.scatter(X, y, color='blue', alpha=0.5, label='данные')

    x_line = np.array([X.min(), X.max()])
    y_line_my = my_model.predict(x_line).reshape(-1, 1)

    x_line = np.array([X.min(), X.max()]).reshape(-1, 1)
    y_line_sk = sk_model.predict(x_line)

    plt.plot(
        x_line,
        y_line_my,
        color='red',
        linewidth=2,
        label=f'my_model: y = {my_model.coef_[0]:.2f}x + {my_model.intercept_:.2f}')

    plt.plot(
        x_line,
        y_line_sk,
        color="#A2D450",
        linewidth=2,
        label=f'sk_model: y = {sk_model.coef_[0]:.2f}x + {sk_model.intercept_:.2f}')


    plt.title('линейная регрессия', fontsize=14)
    plt.xlabel('Признак (X)', fontsize=12)
    plt.ylabel('Ответ (y)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()