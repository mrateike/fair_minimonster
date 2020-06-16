# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging
import numpy as np
import pandas as pd
import pickle
import scipy.optimize as opt
from sklearn.dummy import DummyClassifier
from time import time
from src.fairlearn.reductions._moments.conditional_selection_rate import DemographicParity

from ._constants import _PRECISION, _INDENTATION, _LINE
from src.fairlearn.reductions._bechavod.classifier_family import ClassifierFamily

logger = logging.getLogger(__name__)
import copy


class _Lagrangian:
    """Operations related to the Lagrangian.

    :param X: the training features
    :type X: Array
    :param sensitive_features: the sensitive features to use for constraints
    :type sensitive_features: Array
    :param y: the training labels
    :type y: Array
    :param estimator: the estimator to fit in every iteration of best_h
    :type estimator: an estimator that has a `fit` method with arguments X, y, and sample_weight
    :param constraints: Object describing the parity constraints. This provides the reweighting
        and relabelling
    :type constraints: `fairlearn.reductions.Moment`
    :param eps: allowed constraint violation
    :type eps: float
    :param B:
    :type B:
    :param opt_lambda: indicates whether to optimize lambda during the calculation of the
        Lagrangian; optional with default value True
    :type opt_lambda: bool
    """

    def __init__(self, dataset1, estimator, constraints, eps, B, X=None, sensitive_features=None, y_loss=None):
        opt_lambda = True

        # XA, L, A
        self._dataset1 = dataset1

        self.X_all = dataset1.drop(['sensitive_features', 'label', 'l0', 'l1'], axis = 1)
        _loss1 = dataset1.loc[:, ['l0','l1']]
        _y1 = dataset1.loc[:,['label']]
        _sensitive_features1 = dataset1.loc[:,['sensitive_features']]


        self.constraints = constraints
        self.constraints.load_data1(self.X_all, _y1, sensitive_features=_sensitive_features1)

        self.obj = self.constraints.default_objective()
        self.obj.load_data1(self.X_all, _loss1)

        if X is not None and sensitive_features is not None and y_loss is not None:

            self.X_all = pd.concat([self.X_all, X], axis = 0, ignore_index=True)

            y_loss2 = copy.deepcopy(y_loss)

            # if type(constraints) is DemographicParity:
            self.constraints.load_data(X, y_loss, sensitive_features=sensitive_features)

            self.obj.load_data(X, y_loss2)

            #self.n2 = self.X.shape[0]



        # self.classifier_family.load()

        self.hsH = pd.Series()
        self.errorsH = pd.Series()
        self.gammasH = pd.DataFrame()

        h_family = ClassifierFamily()
        self.classifiersH = h_family.classifiers
        h_idx = 0
        for h in h_family.classifiers:
            self.hsH.at[h_idx] = h
            self.errorsH.at[h_idx] = self.obj.gamma(h)
            self.gammasH[h_idx] = self.constraints.gamma(h)
            h_idx += 1

        self.pickled_estimator = pickle.dumps(estimator)
        self.eps = eps
        self.B = B
        self.opt_lambda = opt_lambda
        self.hs = pd.Series(dtype="float64")
        self.classifiers = pd.Series(dtype="float64")
        self.errors = pd.Series(dtype="float64")
        self.gammas = pd.DataFrame()
        self.lambdas = pd.DataFrame()

        self.n_oracle_calls = 0
        self.n_oracle_calls_dummy_returned = 0
        self.oracle_execution_times = []
        self.last_linprog_n_hs = 0
        self.last_linprog_result = None

    def _eval(self, pi, lambda_vec):
        """Return the value of the Lagrangian.

        :param pi: `pi` is either a series of weights summing up to 1 that indicate the weight of
            each `h` in contributing to the randomized classifier, or a callable corresponding to
            a deterministic predict function.
        :type pi: pandas.Series or callable
        :param lambda_vec: lambda vector
        :type lambda_vec: pandas.Series

        :return: tuple `(L, L_high, gamma, error)` where `L` is the value of the Lagrangian,
            `L_high` is the value of the Lagrangian under the best response of the lambda player,
            `gamma` is the vector of constraint violations, and `error` is the empirical error
        """
        if callable(pi):
            error = self.obj.gamma(pi)[0]
            gamma = self.constraints.gamma(pi)
        else:
            error = self.errors[pi.index].dot(pi)
            gamma = self.gammas[pi.index].dot(pi)

        if self.opt_lambda:
            lambda_projected = self.constraints.project_lambda(lambda_vec)
            L = error + np.sum(lambda_projected * gamma) - self.eps * np.sum(lambda_projected)
        else:
            L = error + np.sum(lambda_vec * gamma) - self.eps * np.sum(lambda_vec)

        max_gamma = gamma.max()
        if max_gamma < self.eps:
            L_high = error
        else:
            L_high = error + self.B * (max_gamma - self.eps)
        return L, L_high, gamma, error

    def eval_gap(self, pi, lambda_hat, nu):
        r"""Return the duality gap object for the given :math:`pi` and :math:`\hat{\lambda}`."""
        L, L_high, gamma, error = self._eval(pi, lambda_hat)
        result = _GapResult(L, L, L_high, gamma, error)
        for mul in [1.0, 2.0, 5.0, 10.0]:
            h_hat, h_hat_idx = self.best_h(mul * lambda_hat)
            # logger.debug("%smul=%.0f", _INDENTATION, mul)
            L_low_mul, _, _, _ = self._eval(pd.Series({h_hat_idx: 1.0}), lambda_hat)
            if L_low_mul < result.L_low:
                result.L_low = L_low_mul
            if result.gap() > nu + _PRECISION:
                break
        return result

    def solve_linprog(self, nu):
        n_hs = len(self.hs)
        n_constraints = len(self.constraints.index)
        if self.last_linprog_n_hs == n_hs:
            return self.last_linprog_result
        c = np.concatenate((self.errors, [self.B]))
        A_ub = np.concatenate((self.gammas - self.eps, -np.ones((n_constraints, 1))), axis=1)
        b_ub = np.zeros(n_constraints)
        A_eq = np.concatenate((np.ones((1, n_hs)), np.zeros((1, 1))), axis=1)
        b_eq = np.ones(1)
        result = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='simplex')
        pi = pd.Series(result.x[:-1], self.hs.index)
        dual_c = np.concatenate((b_ub, -b_eq))
        dual_A_ub = np.concatenate((-A_ub.transpose(), A_eq.transpose()), axis=1)
        dual_b_ub = c
        dual_bounds = [(None, None) if i == n_constraints else (0, None) for i in range(n_constraints + 1)]  # noqa: E501
        result_dual = opt.linprog(dual_c,
                                  A_ub=dual_A_ub,
                                  b_ub=dual_b_ub,
                                  bounds=dual_bounds,
                                  method='simplex')
        lambda_vec = pd.Series(result_dual.x[:-1], self.constraints.index)
        self.last_linprog_n_hs = n_hs
        self.last_linprog_result = (pi, lambda_vec, self.eval_gap(pi, lambda_vec, nu))
        return self.last_linprog_result

    def _call_oracle(self, lambda_vec):

        # without fair learning h
        #signed_weights = self.obj.signed_weights()
        # fair learning h

        # print('err_C:', self.obj.signed_weights())
        # print('gamma_C: ', self.constraints.signed_weights(lambda_vec))

        signed_weights = self.obj.signed_weights() + self.constraints.signed_weights(lambda_vec)
        redY = 1 * (signed_weights > 0)
        redW = signed_weights.abs()

        # Origignal was imposing n as a parameter for reweighting?

        redW = redY.shape[0] * redW / redW.sum()
        # print('call oracle: redY.shape[0]', redY.shape[0])

        # redY_unique = np.unique(redY)
        # classifier = None
        # if len(redY_unique) == 1:
        #     # logger.debug("redY had single value. Using DummyClassifier")
        #     classifier = DummyClassifier(strategy='constant',
        #                                  constant=redY_unique[0])
        #     self.n_oracle_calls_dummy_returned += 1
        # else:


        if len(np.unique(redY))>1:
            classifier = pickle.loads(self.pickled_estimator)
            oracle_call_start_time = time()

            classifier.fit(self.X_all, redY, sample_weight=redW)

            self.oracle_execution_times.append(time() - oracle_call_start_time)
            self.n_oracle_calls += 1
        else:
            classifier = None

        return classifier

    def best_h(self, lambda_vec):
        """Solve the best-response problem.

        Returns the classifier that solves the best-response problem for
        the vector of Lagrange multipliers `lambda_vec`.
        """

        valuesH = self.errorsH + self.gammasH.transpose().dot(lambda_vec)
        best_idxH = valuesH.idxmin()
        h_value = valuesH[best_idxH]
        h = self.hsH.at[best_idxH]
        classifier = self.classifiersH.at[best_idxH]
        h_error = self.errorsH.at[best_idxH]
        h_gamma = self.gammasH.iloc[:, best_idxH]

        classifier_csc = self._call_oracle(lambda_vec)
        if classifier_csc is not None:
            def h_csc(X): return classifier_csc.predict(X)
            h_error_csc = self.obj.gamma(h_csc)
            h_gamma_csc = self.constraints.gamma(h_csc)
            h_value_csc = h_error_csc + h_gamma_csc.dot(lambda_vec)

            if h_value_csc < h_value:
                h_value = h_value_csc
                h = h_csc
                classifier = classifier_csc
                h_error =  h_error_csc
                h_gamma = h_gamma_csc

        if not self.hs.empty:
            values = self.errors + self.gammas.transpose().dot(lambda_vec)
            best_idx = values.idxmin()
            best_value = values[best_idx]
        else:
            best_idx = -1
            best_value = np.PINF

        if h_value < best_value - _PRECISION:
            # logger.debug("%sbest_h: val improvement %f", _LINE, best_value - h_value)
            h_idx = len(self.hs)
            self.hs.at[h_idx] = h
            self.classifiers.at[h_idx] = classifier
            self.errors.at[h_idx] = h_error
            self.gammas[h_idx] = h_gamma
            self.lambdas[h_idx] = lambda_vec.copy()
            best_idx = h_idx

        return self.hs[best_idx], best_idx


class _GapResult:
    """The result of a duality gap computation."""

    def __init__(self, L, L_low, L_high, gamma, error):
        self.L = L
        self.L_low = L_low
        self.L_high = L_high
        self.gamma = gamma
        self.error = error

    def gap(self):
        return max(self.L - self.L_low, self.L_high - self.L)
