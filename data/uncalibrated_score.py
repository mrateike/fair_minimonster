import numpy as np
from scipy.stats.distributions import truncnorm
from numpy.random import RandomState

def get_random(seed=None):
    if seed is None:
        return RandomState()
    else:
        return RandomState(seed)

class BaseDistribution():
    def __init__(self, bias=False):
        self.bias = bias

    @property
    def feature_dimension(self):
        raise NotImplementedError("Subclass must override feature_dimension property.")

    def _sample_test_dataset_core(self, n_test, random):
        raise NotImplementedError("Subclass must override _sample_test_dataset_core(self, n_test).")

    def _sample_train_dataset_core(self, n_train, random, shifts):
        raise NotImplementedError("Subclass must override _sample_train_dataset_core(self, n_train).")

    def sample_test_dataset(self, n_test, shifts, seed=None):
        """
        Draws a nxd matrix of non-sensitive feature vectors, a n-dimensional vector of sensitive attributes
        and a n-dimensional ground truth vector used for testing.

        Args:
            n: The number of examples for which to draw attributes.

        Returns:
            x: nxd matrix of non-sensitive feature vectors
            s: n-dimensional vector of sensitive attributes
        """
        return self._sample_test_dataset_core(n_test, get_random(seed) if seed else get_random(), shifts)

    def sample_train_dataset(self, n_train, shifts, seed=None):
        """
        Draws a nxd matrix of non-sensitive feature vectors, a n-dimensional vector of sensitive attributes
        and a n-dimensional ground truth vector used for training.

        Args:
            n: The number of examples for which to draw attributes.

        Returns:
            x: nxd matrix of non-sensitive feature vectors
            s: n-dimensional vector of sensitive attributes
        """
        return self._sample_train_dataset_core(n_train, get_random(seed) if seed else get_random(), shifts)

class GenerativeDistribution(BaseDistribution):
    def __init__(self, fraction_protected, bias=False):
        super(GenerativeDistribution, self).__init__(bias)
        self.fraction_protected = fraction_protected

    def _sample_features(self, n, fraction_protected, random, shifts):
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

    def _sample_train_dataset_core(self, n_train, random, shifts):
        x, s = self._sample_features(n_train, self.fraction_protected, random, shifts)
        y = self._sample_labels(x, s, random)

        return x, s, y

    def _sample_test_dataset_core(self, n_test, random, shifts):
        return self.sample_train_dataset(n_test, shifts)

class UncalibratedScore(GenerativeDistribution):
    """An distribution modelling an uncalibrated score."""

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
        """Get the probability of repayment."""
        num = (
            np.tan(x)
            + np.tan(self.bound)
            + self.height
            * np.exp(-self.width * (x - self.bound - self.shift) ** 4)
        )
        den = 2 * np.tan(self.bound) + self.height
        return num / den

    def _sample_features(self, n, fraction_protected, random, shifts):

        s = (
                random.rand(n, 1) < fraction_protected
        ).astype(int)

        if shifts is True:
            shifts = s - 0.5
        else:
            shifts = (
                    random.rand(n, 1) < 0
            ).astype(int)

        x = truncnorm.rvs(
            -self.bound + shifts, self.bound + shifts, loc=-shifts
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

