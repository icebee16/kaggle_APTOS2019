from functools import partial

import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.optimize import minimize


class OptimizedQWK(object):
    """
    reference
    ---------
    https://www.kaggle.com/abhishek/optimizer-for-quadratic-weighted-kappa
    """
    def __init__(self):
        self.init_coef()

    def init_coef(self):
        self._coef = [0.5, 1.5, 2.5, 3.5]

    def set_coef(self, coef):
        self._coef = coef

    def get_coef(self):
        return self._coef

    def discretization(self, y_continuous, coef=None):
        y_discrete = np.copy(y_continuous)
        if coef is None:
            coef = self._coef

        for i, pred in enumerate(y_continuous):
            if pred < coef[0]:
                y_discrete[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                y_discrete[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                y_discrete[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                y_discrete[i] = 3
            elif pred >= coef[3]:
                y_discrete[i] = 4
        return y_discrete

    def __inv_kappa_loss(self, coef, y_true, y_pred):
        """
        Calc inverse kappa loss
        Inverse score for minimize optim
        """
        _y_pred = self.discretization(y_pred, coef=coef)
        score = cohen_kappa_score(y_true.reshape(-1),
                                  _y_pred.reshape(-1),
                                  labels=[0, 1, 2, 3, 4],
                                  weights="quadratic")
        return -score

    def fit(self, y_true, y_pred):
        loss_partial = partial(self.__inv_kappa_loss, y_true=y_true, y_pred=y_pred)
        self._coef = minimize(loss_partial, self._coef, method="nelder-mead")["x"]


if __name__ == "__main__":
    estimator = OptimizedQWK()
    estimator.set_coef([0.48, 1.5, 2.5, 3.5])
    estimator.set_coef([0.48, 1.5, 2.5, 3.5])

    a = np.array([0.2, 0.49, 2.4, 4.2, 0.5001, 0.48, 0.48, 0.48, 0.2, 0.49, 2.4, 4.2, 0.5001, 0.48, 0.48, 0.48]).reshape((-1, 1))
    b = np.array([0, 1, 2, 4, 1, 0, 0, 0, 0, 1, 2, 4, 1, 0, 0, 0]).reshape((-1, 1))

    print(b)
    print(estimator.get_coef())
    print(estimator.discretization(a))
    estimator.fit(b, a)
    print(estimator.get_coef())
    print(estimator.discretization(a))
