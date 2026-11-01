import numpy as np


class BaseEstimator:
    def get_params(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.endswith('_')
        }


class ClassifierMixin:
    _estimator_type = 'classifier'

    def score(self, X, y):
        from .metrics import accuracy

        return accuracy(y, self.predict(X))


class RegressorMixin:
    _estimator_type = 'regressor'

    def score(self, X, y):
        from .metrics import r2_score

        return r2_score(y, self.predict(X))