import numpy as np
from abc import ABC, abstractmethod


class BaseEstimator(ABC):
    def get_params(self):
            return {
                key: value
                for key, value in self.__dict__.items()
                if not key.endswith('_')
            }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class ClassifierMixin(ABC):
    _estimator_type = 'classifier'

    def score(self, X, y):
        from .metrics import accuracy

        return accuracy(y, self.predict(X))


class RegressorMixin(ABC):
    _estimator_type = 'regressor'

    def score(self, X, y):
        from .metrics import r2_score

        return r2_score(y, self.predict(X))