import numpy as np
from scipy.optimize import minimize_scalar

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
        criterion = 'gini'
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = criterion
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
                max_features=self.max_features,
                criterion=self.criterion
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


class GradientBoostingRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        n_estimators: int = 100,
        lr: float = 0.1,
        max_depth: int = 4,
        min_samples_split: int = 2,
        min_samples_leaf: int = 5,
        max_features = None,
        loss: str = 'mse'
    ):
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.loss = loss
        self.estimators_ = []
        self.base_pred_ = None

    def _create_tree(self):

        return DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features
        )

    def _calc_residuals(self, y, y_pred):
        if self.loss == 'mse':
            return y - y_pred
        elif self.loss == 'mae':
            return np.sign(y - y_pred)
        else:
            raise ValueError('неправильная loss функция')

    def _find_gamma(self, y, y_pred, tree_pred):
        if self.loss == 'mse':
            return 1.0
            
        elif self.loss == 'mae':
            residual = y - y_pred

            def loss_func(gamma):
                return np.sum(np.abs(residual - gamma * tree_pred))
            
            res = minimize_scalar(loss_func)
            return res.x

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        if self.loss == 'mse':
            self.base_pred_ = np.mean(y)
        elif self.loss == 'mae':
            self.base_pred_ = np.median(y)
        else:
            raise ValueError('неправильная loss функция')

        curr_pred = np.full(shape=y.shape, fill_value=self.base_pred_)

        for _ in range(self.n_estimators):
            residuals = self._calc_residuals(y, curr_pred)

            tree = self._create_tree()
            tree.fit(X, residuals)
            tree_pred = tree.predict(X)

            gamma = self._find_gamma(y, curr_pred, tree_pred)
            curr_pred += self.lr * gamma * tree_pred

            self.estimators_.append(tree, gamma)

        return self

    def predict(self, X):
        X = np.asarray(X)
        y_pred = np.full(shape=X.shape[0], fill_value=self.base_pred_)

        for tree, gamma in self.estimators_:
            y_pred += self.lr * gamma * tree.predict(X)

        return y_pred


class GradientBoostingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 4,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features=None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.estimators_ = []
        self.base_pred_ = None
        self.classes_ = None

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def _create_tree(self):
        return DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
        )

    def _find_gamma(self, y, p, tree_pred):
        num = np.sum((y - p) * tree_pred)
        den = np.sum(p * (1.0 - p) * (tree_pred ** 2))

        return num / den

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        y_binary = np.where(y == self.classes_[1], 1.0, 0.0)

        p_mean = np.mean(y_binary)
        self.base_pred_ = np.log(p_mean / (1.0 - p_mean))

        pred = np.full(shape=X.shape[0], fill_value=self.base_pred_)

        for _ in range(self.n_estimators):
            p = self._sigmoid(pred)
            residuals = y_binary - p

            tree = self._create_tree()
            tree.fit(X, residuals)
            tree_pred = tree.predict(X)

            gamma = self._find_gamma(y_binary, p, tree_pred)
            pred += self.learning_rate * gamma * tree_pred

            self.estimators_.append((tree, gamma))

        return self

    def predict_raw(self, X):
        X = np.asarray(X)
        raw_pred = np.full(shape=X.shape[0], fill_value=self.base_pred_)

        for tree, gamma in self.estimators_:
            raw_pred += self.learning_rate * gamma * tree.predict(X)

        return raw_pred

    def predict_proba(self, X):
        raw_pred = self.predict_raw(X)
        p1 = self._sigmoid(raw_pred)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return np.where(proba >= 0.5, self.classes_[1], self.classes_[0])