import pandas as pd
from src.contextual_bandit.Policy import *

from data.distribution import UncalibratedScore, FICODistribution, AdultCreditDistribution, COMPASDataset


class DatasetBandit(object):

    def __init__(self, distribution):
        self.fraction_protected = 0.5
        self.test_percentage = 0.2

        if distribution == 'Uncalibrated':
            self.distribution = UncalibratedScore(self.fraction_protected)
        elif distribution == 'FICO':
            self.distribution = FICODistribution(self.fraction_protected)
        elif distribution == 'Adult':
            self.distribution = AdultCreditDistribution(self.test_percentage)
        elif distribution == 'COMPAS':
            self.distribution = COMPASDistribution(self.fraction_protected)
        else:
            print('SIMULATOR ERROR')

    def sample_test_dataset(self, T, seed):
        x, a, y = self.distribution.sample_train_dataset(T, seed)
        X = pd.Series(x.squeeze(), name='features')
        Y = pd.Series(y.squeeze(), name='label').astype(int)
        A = pd.Series(a.squeeze(), name='sensitive_features_X').astype(int)
        XA = pd.concat([X, A == 1], axis=1).astype(float)
        # XA = pd.concat([X, A], axis=1).astype(float)
        A = A.rename('sensitive_features')

        return XA, A, Y

    def sample_dataset(self, T, seed):
        x, a, y = self.distribution.sample_train_dataset(T, seed)
        X = pd.Series(x.squeeze(), name='features')
        Y = pd.Series(y.squeeze(), name='label').astype(int)
        A = pd.Series(a.squeeze(), name='sensitive_features_X').astype(int)
        XA = pd.concat([X, A == 1], axis=1).astype(float)
        # XA = pd.concat([X, A], axis=1).astype(float)
        A = A.rename('sensitive_features')


        L = pd.DataFrame(columns=['l0', 'l1'])

        for i, value in Y.items():
            if value == 0:
                L.at[i, 'l0'] = 0
                L.at[i, 'l1'] = 1
            elif value == 1:
                L.at[i, 'l0'] = 1
                L.at[i, 'l1'] = 0
            else:
                print('ERROR: Simulator')
        dataset = pd.concat([XA, L, A, Y], axis=1)
        return dataset


    def get_loss(self, d, y):
        # y is pd.Series
        # d is int

        if d == 0:
            loss = 0.5
        else:
            if y == 0:
                # d = 1, y = 0
                loss = 1
            else:
                # d = 1, y = 1
                loss = 0
        return loss




