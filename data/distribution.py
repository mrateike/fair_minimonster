# from Floyd

import os
import sys

root_path = os.path.abspath(os.path.join('.'))
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np
# pylint: disable=no-name-in-module
from scipy.special import expit as sigmoid
from scipy.stats.distributions import truncnorm
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
from data.util import train_test_split, get_random, whiten
from responsibly.dataset import build_FICO_dataset, COMPASDataset, AdultDataset, GermanDataset


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


class SplitDistribution(GenerativeDistribution):
    def __init__(self, fraction_protected, bias=False):
        super(SplitDistribution, self).__init__(fraction_protected=fraction_protected, bias=bias)

    @property
    def feature_dimension(self):
        return 2 if self.bias else 1

    def _sample_features(self, n, fraction_protected, random):
        s = (
                random.rand(n, 1) < fraction_protected
        ).astype(int)
        x = 3.5 * random.randn(n, 1) + 3 * (0.5 - s)

        if self.bias:
            ones = np.ones((n, 1))
            x = np.hstack((ones, x))

        return x, s

    def _sample_labels(self, x, s, random):
        if self.bias:
            x = x[:, 1]

        yprob = 0.8 * sigmoid(0.6 * (x + 3)) * sigmoid(
            -5 * (x - 3)
        ) + sigmoid(x - 5)

        return np.expand_dims(random.binomial(1, yprob), axis=1)


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

    def _sample_features(self, n, fraction_protected, random):
        s = (
                random.rand(n, 1) < fraction_protected
        ).astype(int)

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
    def __init__(self, fraction_protected, bias=False):
        super(FICODistribution, self).__init__(fraction_protected=fraction_protected, bias=bias)
        self.fico_data = build_FICO_dataset()
        self.precision = 4

    @property
    def feature_dimension(self):
        return 2 if self.bias else 1

    def _sample_features(self, n, fraction_protected, random):
        fico_cdf = self.fico_data["cdf"]

        unprotected_cdf = fico_cdf["White"].values
        protected_cdf = fico_cdf["Black"].values
        shifted_scores = (fico_cdf.index.values * 10) + 10

        s = (
                random.rand(n, 1) < fraction_protected
        ).astype(int).squeeze()

        s0_idx = np.where(s == 0)[0]
        s1_idx = np.where(s == 1)[0]

        rands_s0 = random.randint(0, 10 ** self.precision, len(s0_idx)) / float(10 ** self.precision)
        rands_s1 = random.randint(0, 10 ** self.precision, len(s1_idx)) / float(10 ** self.precision)

        previous_unprotected_threshold = -1
        previous_protected_threshold = -1

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


class ResamplingDistribution(BaseDistribution):
    """Resample from a finite dataset."""

    def _load_data(self):
        raise NotImplementedError("Subclass must override _load_data(self).")

    def __init__(self, test_percentage, bias=False):
        super(ResamplingDistribution, self).__init__(bias)
        x, s, y = self._load_data()
        self.x, self.x_test, self.y, self.y_test, self.s, self.s_test = train_test_split(x, y, s,
                                                                                         test_size=test_percentage)
        self.total_test_samples = self.x_test.shape[0]
        self.test_sample_indices = np.arange(self.total_test_samples)

        self.total_training_samples = self.x.shape[0]
        self.training_sample_indices = np.arange(self.total_training_samples)

        if self.bias:
            self.x = np.hstack([np.ones([self.total_training_samples, 1]), self.x])
            self.x_test = np.hstack(
                [np.ones([self.x_test.shape[0], 1]), self.x_test]
            )
        self.feature_dim = self.x.shape[1]

    @property
    def feature_dimension(self):
        return self.x_test.shape[1]

    def _sample_train_dataset_core(self, n_train, random):
        n = min(self.total_training_samples, n_train)
        indices = random.choice(self.training_sample_indices, n, replace=True)

        x = self.x[indices].reshape((n, -1))
        y = self.y[indices].reshape((n, -1))
        s = self.s[indices].reshape((n, -1))

        # n_train_remain = max(n_train - self.total_training_samples, 0)
        # while n_train_remain > 0:
        #     n = min(self.total_training_samples, n_train_remain)
        #     indices = random.choice(self.training_sample_indices, n, replace=True)
        #
        #     x = np.vstack((x, self.x[indices].reshape((n, -1))))
        #     y = np.vstack((y, self.y[indices].reshape((n, -1))))
        #     s = np.vstack((s, self.s[indices].reshape((n, -1))))
        #
        #     n_train_remain = max(n_train_remain - self.total_training_samples, 0)

        return x, s, y

    def _sample_test_dataset_core(self, n_test, random):
        n = min(self.total_test_samples, n_test) if n_test is not None else self.total_test_samples
        indices = random.choice(self.test_sample_indices, n, replace=True)

        x = self.x_test[indices].reshape((n, -1))
        y = self.y_test[indices].reshape((n, -1))
        s = self.s_test[indices].reshape((n, -1))

        return x, s, y


class COMPASDistribution(ResamplingDistribution):
    def __init__(self, test_percentage, bias=False):
        super(COMPASDistribution, self).__init__(test_percentage, bias)

    def _load_data(self):
        compas_data = COMPASDataset()

        # use race as the sensitive attribute
        race = compas_data.df['race']
        s = race.where(race == 'Caucasian', 1)
        s.where(s == 1, 0, inplace=True)
        s = s.values.reshape(-1, 1)

        # Use juvenile felonies, juvenile misdemeanors, juvenile others, prior conviction
        x = compas_data.df[['juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count']].values

        # Charge Degree categories in one hot encoding
        for category in compas_data.df['c_charge_degree'].unique():
            degree_category = compas_data.df['c_charge_degree'].where(compas_data.df['c_charge_degree'] == category, 0)
            degree_category.where(degree_category == 0, 1, inplace=True)
            x = np.hstack((x, degree_category.values.reshape(-1, 1)))

        # use actual recidivisim as target variable
        y = compas_data.df[compas_data.target].values.reshape(-1, 1)

        return x.astype(float), s.astype(float), y.astype(float)


class AdultCreditDistribution(ResamplingDistribution):
    def __init__(self, test_percentage, bias=False):
        super(AdultCreditDistribution, self).__init__(test_percentage, bias)

    def _load_data(self):
        data = AdultDataset()

        print('length Adult', len(data))

        # use race as the sensitive attribute
        race = data.df['race']
        s = race.where(race == 'White', 1)
        s.where(s == 1, 0, inplace=True)
        s = s.values.reshape(-1, 1)

        # Use capital gain/capital loss and hours per week
        x = whiten(data=data.df[['capital_gain', 'capital_loss', 'hours_per_week']].values.astype(float))

        # work class, education, marital status and native country in one hot encoding
        for column in ["workclass", "education", "marital_status", "native_country"]:
            for category in data.df[column].unique():
                category = data.df[column].where(data.df[column] == category, 0)
                category.where(category == 0, 1, inplace=True)
                x = np.hstack((x, category.values.reshape(-1, 1)))

        # use actual income as target variable: >50K = 1, <=50K = 0
        income = data.df[data.target]
        y = income.where(income == '>50K', 0)
        y.where(y == 0, 1, inplace=True)
        y = y.values.reshape(-1, 1)

        return x.astype(float), s.astype(float), y.astype(float)


class GermanCreditDistribution(ResamplingDistribution):
    def __init__(self, test_percentage, bias=False):
        super(GermanCreditDistribution, self).__init__(test_percentage, bias)

    def _load_data(self):
        data = GermanDataset()

        # use sex as the sensitive attribute (columns status and sex are switched)
        sex = data.df['status'].iloc[:, 0]
        s = sex.where(sex == 'male', 1)
        s.where(s == 1, 0, inplace=True)
        s = s.values.reshape(-1, 1)

        # Use credit amount, installment rate, time in present residence, number of existing credits, number of people
        # liable for and whether or not person is a foreign worker
        x = whiten(data.df[['credit_amount',
                            'installment_rate',
                            'present_residence_since',
                            'number_of_existing_credits',
                            'number_of_people_liable_for']].values.astype(float))

        # credit history, purpose of credit, savings, length of current employment, property, housing situation
        # and current job type in one hot encoding
        for column in ["credit_history", "purpose", "savings", "present_employment", "property", "housing", "job"]:
            for category in data.df[column].unique():
                category = data.df[column].where(data.df[column] == category, 0)
                category.where(category == 0, 1, inplace=True)
                x = np.hstack((x, category.values.reshape(-1, 1)))

        # was given credit good or bad
        credit = data.df[data.target]
        y = credit.where(credit == 'good', 0)
        y.where(y == 0, 1, inplace=True)
        y = y.values.reshape(-1, 1)

        return x.astype(float), s.astype(float), y.astype(float)
