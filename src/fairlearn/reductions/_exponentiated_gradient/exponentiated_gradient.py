# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from ._constants import _ACCURACY_MUL, _REGRET_CHECK_START_T, _REGRET_CHECK_INCREASE_T, \
    _SHRINK_REGRET, _SHRINK_ETA, _MIN_T, _RUN_LP_STEP, _PRECISION, _INDENTATION
from ._lagrangian import _Lagrangian
from src.fairlearn._input_validation import _KW_SENSITIVE_FEATURES
from src.fairlearn.reductions._moments.conditional_selection_rate import ClassificationMoment

logger = logging.getLogger(__name__)


class ExponentiatedGradient(BaseEstimator, MetaEstimatorMixin):
    """An Estimator which implements the exponentiated gradient approach to reductions.

    The exponentiated gradient algorithm is described in detail by
    `Agarwal et al. (2018) <https://arxiv.org/abs/1803.02453>`_.

    :param estimator: An estimator implementing methods :code:`fit(X, y, sample_weight)` and
        :code:`predict(X)`, where `X` is the matrix of features, `y` is the vector of labels, and
        `sample_weight` is a vector of weights; labels `y` and predictions returned by
        :code:`predict(X)` are either 0 or 1.
    :type estimator: estimator

    :param constraints: The disparity constraints expressed as moments
    :type constraints: fairlearn.reductions.Moment

    :param eps: Allowed fairness constraint violation; the solution is guaranteed to have the
        error within :code:`2*best_gap` of the best error under constraint eps; the constraint
        violation is at most :code:`2*(eps+best_gap)`
    :type eps: float

    :param T: Maximum number of iterations
    :type T: int

    :param nu: Convergence threshold for the duality gap, corresponding to a
        conservative automatic setting based on the statistical uncertainty in measuring
        classification error
    :type nu: float

    :param eta_mul: Initial setting of the learning rate
    :type eta_mul: float
    """

    def __init__(self, dataset1, estimator, constraints, eps=0.01, T=50, nu=None, eta0=2.0):  # noqa: D103
        self._estimator = estimator
        self._constraints = constraints
        self._eps = eps
        self._T = T
        self._nu = nu
        self._eta0 = eta0
        self._dataset1 = dataset1
        self.B = 2

        self._best_gap = None
        self._predictors = None
        self._weights = None
        self._last_t = None
        self._best_t = None
        self._n_oracle_calls = 0
        self._n_oracle_calls_dummy_returned = 0
        self._oracle_execution_times = None
        self._lambda_vecs = pd.DataFrame()
        self._lambda_vecs_LP = pd.DataFrame()
        self._lambda_vecs_lagrangian = pd.DataFrame()

    def fit(self, X=None, y=None, **kwargs):
        """Return a fair classifier under specified fairness constraints.

        :param X: The feature matrix
        :type X: numpy.ndarray or pandas.DataFrame

        :param y: The label vector
        :type y: numpy.ndarray, pandas.DataFrame, pandas.Series, or list
        """
        # if isinstance(self._constraints, ClassificationMoment):
        #     logger.debug("Classification problem detected")
        #     is_classification_reduction = True
        # else:
        #     logger.debug("Regression problem detected")
        #     is_classification_reduction = False

        # doesnt work because y has two columns
        # _, y_train, sensitive_features = _validate_and_reformat_input(
        #     X, y, enforce_binary_labels=is_classification_reduction, **kwargs)
        # n = y_train.shape[0]

        self.lambda_vecs_EG_ = pd.DataFrame()
        self.lambda_vecs_LP_ = pd.DataFrame()
        self.lambda_vecs_ = pd.DataFrame()


        if y is not None and X is not None:
            if _KW_SENSITIVE_FEATURES in kwargs:
                sensitive_features = kwargs[_KW_SENSITIVE_FEATURES]
            y_train = y
            #n = y_train.shape[0]
            lagrangian = _Lagrangian(self._dataset1, self._estimator,
                                     self._constraints, self._eps, self.B, X, sensitive_features, y_train)

        else:
            lagrangian = _Lagrangian(self._dataset1, self._estimator,
                                     self._constraints, self._eps, self.B)

        #B = 1 / self._eps[0]
        #(1/0.3 = 3.33, 1/0.01 = 100) / 1/0.5 = 2
        # Assumption sens_att identity functions in H: B = 2


        theta = pd.Series(0, lagrangian.constraints.index)
        pisum = pd.Series(dtype="float64")
        gaps_EG = []
        gaps = []
        pis = []

        last_regret_checked = _REGRET_CHECK_START_T
        last_gap = np.PINF
        for t in range(0, self._T):
            #logger.debug("...iter=%03d", t)

            # set lambdas for every constraint
            lambda_vec = self.B * np.exp(theta) / (1 + np.exp(theta).sum())
            self.lambda_vecs_EG_[t] = lambda_vec
            lambda_EG = self.lambda_vecs_EG_.mean(axis=1)

            # select classifier according to best_h method
            h, h_idx = lagrangian.best_h(lambda_vec)

            if t == 0:
                # if self._nu is None:
                #     self._nu = _ACCURACY_MUL * (h(X) - y_train).abs().std() / np.sqrt(n)
                #eta_min = self._nu / (2 * B)
                # eta is 1
                eta = self._eta0 / self.B

            if h_idx not in pisum.index:
                pisum.at[h_idx] = 0.0
            pisum[h_idx] += 1.0
            gamma = lagrangian.gammas[h_idx]
            pi_EG = pisum / pisum.sum()
            result_EG = lagrangian.eval_gap(pi_EG, lambda_EG, self._nu)
            gap_EG = result_EG.gap()
            gaps_EG.append(gap_EG)

            if t == 0 or not _RUN_LP_STEP:
                gap_LP = np.PINF
            else:
                # saddle point optimization over the convex hull of
                # classifiers returned so far
                pi_LP, self.lambda_vecs_LP_[t], result_LP = lagrangian.solve_linprog(self._nu)
                gap_LP = result_LP.gap()

            # keep values from exponentiated gradient or linear programming
            if gap_EG < gap_LP:
                pis.append(pi_EG)
                gaps.append(gap_EG)
            else:
                pis.append(pi_LP)
                gaps.append(gap_LP)

            # logger.debug("%seta=%.6f, L_low=%.3f, L=%.3f, L_high=%.3f, gap=%.6f, disp=%.3f, "
            #              "err=%.3f, gap_LP=%.6f",
            #              _INDENTATION, eta, result_EG.L_low, result_EG.L, result_EG.L_high,
            #              gap_EG, result_EG.gamma.max(), result_EG.error, gap_LP)

            if (gaps[t] < self._nu) and (t >= _MIN_T):
                # solution found
                break

            # update regret (_REGRET_CHECK_INCREASE_T = 1.6, _SHRINK_REGRET = 0.8
            # _SHRINK_ETA = 0.8)
            if t >= last_regret_checked * _REGRET_CHECK_INCREASE_T:
                best_gap = min(gaps_EG)

                if best_gap > last_gap * _SHRINK_REGRET:
                    eta *= _SHRINK_ETA
                last_regret_checked = t
                last_gap = best_gap

            # update theta based on learning rate
            theta += eta * (gamma - self._eps)

        # retain relevant result data
        gaps_series = pd.Series(gaps)
        gaps_best = gaps_series[gaps_series <= gaps_series.min() + _PRECISION]
        self._best_t = gaps_best.index[-1]
        self._best_gap = gaps[self._best_t]
        self._weights = pis[self._best_t]
        self._hs = lagrangian.hs
        for h_idx in self._hs.index:
            if h_idx not in self._weights.index:
                self._weights.at[h_idx] = 0.0

        self._last_t = len(pis) - 1
        self._predictors = lagrangian.classifiers
        self._n_oracle_calls = lagrangian.n_oracle_calls
        self._n_oracle_calls_dummy_returned = lagrangian.n_oracle_calls_dummy_returned
        self._oracle_execution_times = lagrangian.oracle_execution_times
        self.lambda_vecs_lagrangian = lagrangian.lambdas

        # logger.debug("...eps=%.3f, B=%.1f, nu=%.6f, T=%d, eta_min=%.6f",
        #              self._eps, B, self._nu, self._T, eta_min)
        # logger.debug("...last_t=%d, best_t=%d, best_gap=%.6f, n_oracle_calls=%d, n_hs=%d",
        #              self._last_t, self._best_t, self._best_gap, lagrangian.n_oracle_calls,
        #              len(lagrangian.classifiers))

    def predict(self, X):
        """Provide a prediction for the given input data.

        Note that this is non-deterministic, due to the nature of the
        exponentiated gradient algorithm.

        :param X: Feature data
        :type X: numpy.ndarray or pandas.DataFrame

        :return: The prediction. If `X` represents the data for a single example
            the result will be a scalar. Otherwise the result will be a vector
        :rtype: Scalar or vector
        """

        probs = self._pmf_predict(X)
        positive_probs = probs[:, 1]

        # print('pmf_predict: positive_probs', positive_probs)
        threshold = np.random.rand(len(positive_probs))
        # print('pmf_predict: threshold', threshold)
        dec = (positive_probs >= threshold) * 1
        # print('pmf_predict: dec', dec)
        i=0
        dec_prob = np.array([[7,7]])

        for d in dec:
            prop_dec = probs[i,int(d)]
            dec_prob = np.append(dec_prob, np.array([[int(d), float(prop_dec)]]), axis=0)
        dec_prob =  dec_prob[1:, :]



        return dec_prob

    def _pmf_predict(self, X):

        """Probability mass function for the given input data.

        :param X: Feature data
        :type X: numpy.ndarray or pandas.DataFrame
        :return: Array of tuples with the probabilities of predicting 0 and 1.
        :rtype: pandas.DataFrame
        """

        pred = pd.DataFrame()
        for t in range(len(self._hs)):
            pred[t] = self._hs[t](X)
        # print('pmf_pred, self._weights', pred, self._weights)
        positive_probs = pred[self._weights.index].dot(self._weights).to_frame()
        return np.concatenate((1-positive_probs, positive_probs), axis=1)
