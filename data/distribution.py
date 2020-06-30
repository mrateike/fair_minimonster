
import os
import sys

root_path = os.path.abspath(os.path.join('.'))
if root_path not in sys.path:
    sys.path.append(root_path)
import numpy as np
from scipy.special import expit as sigmoid
from scipy.stats.distributions import truncnorm
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
from data.util import train_test_split, get_random, whiten
from responsibly.dataset import build_FICO_dataset


"""
(c) Floyd Kretschmar (https://github.com/floydkretschmar/master-thesis)
Specifies and generates datasets to be sampled from 
Options: UncalibratedScore (synthetic data), FICODistribution
"""


class BaseDistribution(object):
    def __init__(self, bias=False):
        self.bias = bias

    @property
    def feature_dimension(self):
        raise NotImplementedError("Subclass must override feature_dimension property.")

    def _sample_test_dataset_core(self, n_test, random):
        raise NotImplementedError("Subclass must override _sample_test_dataset_core(self, n_test).")

    def _sample_train_dataset_core(self, n_train, random):
        raise NotImplementedError("Subclass must override _sample_train_dataset_core(self, n_train).")

    def sample_test_dataset(self, n_test, seed=None):
        """
         Draws a nxd matrix of non-sensitive feature vectors, a n-dimensional vector of sensitive attributes
         and a n-dimensional ground truth vector used for testing.
         Args:
             n: The number of examples for which to draw attributes.
         Returns:
             x: nxd matrix of non-sensitive feature vectors
             s: n-dimensional vector of sensitive attributes
         """
        return self._sample_test_dataset_core(n_test, get_random(seed) if seed else get_random())

    def sample_train_dataset(self, n_train, seed=None):
        """
         Draws a nxd matrix of non-sensitive feature vectors, a n-dimensional vector of sensitive attributes
         and a n-dimensional ground truth vector used for training.
         Args:
             n: The number of examples for which to draw attributes.
         Returns:
             x: nxd matrix of non-sensitive feature vectors
             s: n-dimensional vector of sensitive attributes
         """
        return self._sample_train_dataset_core(n_train, get_random(seed) if seed else get_random())


class GenerativeDistribution(BaseDistribution):
    def __init__(self, fraction_protected, bias=False):
        super(GenerativeDistribution, self).__init__(bias)
        self.fraction_protected = fraction_protected

    def _sample_features(self, n, fraction_protected, random):
        """
        Draws both a nxd matrix of non-sensitive feature vectors, as well as a n-dimensional vector
        of sensitive attributes.
        Args:
            n: The number of examples for which to draw attributes.
        Returns:
            x: nxd matrix of non-sensitive feature vectors
            s: n-dimensional vector of sensitive attributes
        """
        raise NotImplementedError("Subclass must override sample_features(self, n).")

    def _sample_labels(self, x, s, random):
     """
    Draws a n-dimensional ground truth vector.
    Args:
        x: nxd matrix of non-sensitive feature vectors
        s: n-dimensional vector of sensitive attributes
    Returns:
        y: n-dimensional ground truth vector
    """
        raise NotImplementedError("Subclass must override sample_labels(self, x, s).")

    def _sample_train_dataset_core(self, n_train, random):
        x, s = self._sample_features(n_train, self.fraction_protected, random)
        y = self._sample_labels(x, s, random)

        return x, s, y

    def _sample_test_dataset_core(self, n_test, random):
        return self.sample_train_dataset(n_test)


class UncalibratedScore(GenerativeDistribution):
    """An distribution modelling an uncalibrated score from
    Kilbertus, N., Rodriguez, M. G., Schölkopf, B., Muandet, K., & Valera, I. (2020, June).
    Fair decisions despite imperfect predictions. In International Conference on Artificial
    Intelligence and Statistics (pp. 277-287).
    http://proceedings.mlr.press/v108/kilbertus20a/kilbertus20a.pdf"""


    @property
    def feature_dimension(self):
        return 2 if self.bias else 1

    def __init__(self, fraction_protected, bias=False):
        super(UncalibratedScore, self).__init__(fraction_protected=fraction_protected, bias=bias)
        self.bound = 0.8
        self.width = 30.0
        self.height = 3.0
        self.shift = 0.1
        self.bias = bias

    def _pdf(self, x):
        """Get the probability of repayment. Uncalibrated monotonous function."""
        num = (
                np.tan(x)
                + np.tan(self.bound)
                + self.height
                * np.exp(-self.width * (x - self.bound - self.shift) ** 4)
        )
        den = 2 * np.tan(self.bound) + self.height
        return num / den

    def _sample_features(self, n, fraction_protected, random):
        """Get senstitive attribute s and non-sentive features x """

        # Get senstitive attribute s from binomial distribution"""
        s = (
                random.rand(n, 1) < fraction_protected
        ).astype(int)

        # Get non-sensitive attribute x from truncated normal distribution
        # with mean dependent on sensitive attribute
        shifts = s - 0.5
        x = truncnorm.rvs(
            -self.bound + shifts, self.bound + shifts, loc=-shifts, random_state = random
        ).reshape(-1, 1)

        if self.bias:
            ones = np.ones((n, 1))
            x = np.hstack((ones, x))

        return x, s

    def _sample_labels(self, x, s, random):
        if self.bias:
            x = x[:, 1]

        yprob = self._pdf(x)
        return np.expand_dims(random.binomial(1, yprob), axis=1)


class FICODistribution(GenerativeDistribution):
    """FICO dataset imported from Responsibly:
    Toolkit for Auditing and Mitigating Bias
    and Fairness of Machine Learning Systems (c) 2018 Shlomi Hod
    https://docs.responsibly.ai/dataset.html#fico-dataset"""


    def __init__(self, fraction_protected, bias=False):
        super(FICODistribution, self).__init__(fraction_protected=fraction_protected, bias=bias)
        self.fico_data = build_FICO_dataset()
        self.precision = 4

    @property
    def feature_dimension(self):
        return 2 if self.bias else 1

    def _sample_features(self, n, fraction_protected, random):
        """sample sensitive-features s and non-sensitive features x """

        fico_cdf = self.fico_data["cdf"]

        # getting binary sensitive attributes
        unprotected_cdf = fico_cdf["White"].values
        protected_cdf = fico_cdf["Black"].values
        shifted_scores = (fico_cdf.index.values * 10) + 10

        # Get senstitive attribute s from binomial distribution"""
        s = (
                random.rand(n, 1) < fraction_protected
        ).astype(int).squeeze()

        s0_idx = np.where(s == 0)[0]
        s1_idx = np.where(s == 1)[0]

        rands_s0 = random.randint(0, 10 ** self.precision, len(s0_idx)) / float(10 ** self.precision)
        rands_s1 = random.randint(0, 10 ** self.precision, len(s1_idx)) / float(10 ** self.precision)

        previous_unprotected_threshold = -1
        previous_protected_threshold = -1

        # Get non-sensitive feature x """
        for score_idx, shifted_score in enumerate(shifted_scores):
            unprotected_threshold = unprotected_cdf[score_idx]
            protected_threshold = protected_cdf[score_idx]

            rands_s0[(rands_s0 > previous_unprotected_threshold) & (rands_s0 <= unprotected_threshold)] = shifted_score
            rands_s1[(rands_s1 > previous_protected_threshold) & (rands_s1 <= protected_threshold)] = shifted_score

            previous_unprotected_threshold = unprotected_threshold
            previous_protected_threshold = protected_threshold

        x = np.zeros(n)
        x[s0_idx] = rands_s0
        x[s1_idx] = rands_s1
        x = ((x - 10) / 1000).reshape(-1, 1)

        if self.bias:
            ones = np.ones((n, 1))
            x = np.hstack((ones, x))

        return x, s.reshape(-1, 1)

    def _sample_labels(self, x, s, random):
        """sample ground truth labels y """

        if self.bias:
            x = x[:, 1]

        y = np.full(x.shape[0], -1)
        local_s = s.squeeze()

        scores = self.fico_data["performance"]["Black"].index.values / 100.0
        non_defaulters_protected = self.fico_data["performance"]["Black"].values
        non_defaulters_unprotected = self.fico_data["performance"]["White"].values

        for idx, score in enumerate(scores):
            x_current_score_idx = np.where(x == score)[0]
            s_current_score = local_s[x_current_score_idx]

            x_s0_current_score_idx = x_current_score_idx[s_current_score == 0]
            x_s1_current_score_idx = x_current_score_idx[s_current_score == 1]

            y_unprotected = (
                    random.rand(len(x_s0_current_score_idx), 1) < non_defaulters_unprotected[idx]
            ).astype(int).squeeze()
            y_protected = (
                    random.rand(len(x_s1_current_score_idx), 1) < non_defaulters_protected[idx]
            ).astype(int).squeeze()

            y[x_s0_current_score_idx] = y_unprotected
            y[x_s1_current_score_idx] = y_protected

        return y.reshape(-1, 1)

