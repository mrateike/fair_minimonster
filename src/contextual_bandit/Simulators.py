# Copyright (c) 2016 akshaykr, adapted by mrateike
import pandas as pd
from src.contextual_bandit.Policy import *

from data.distribution import UncalibratedScore, FICODistribution

"""
Adapted from akshaykr (https://github.com/akshaykr/oracle_cb)
This arranges test and training data to pd.DataFrames
and the transformed loss function
"""

class DatasetBandit(object):

    def __init__(self, distribution):
        self.fraction_protected = 0.5

        if distribution == 'Uncalibrated':
            self.distribution = UncalibratedScore(self.fraction_protected)
        elif distribution == 'FICO':
            self.distribution = FICODistribution(self.fraction_protected)
        else:
            print('SIMULATOR ERROR')

    def sample_test_dataset(self, T, seed):
        x, a, y = self.distribution.sample_train_dataset(T, seed)
        X = pd.Series(x.squeeze(), name='features')
        Y = pd.Series(y.squeeze(), name='label').astype(int)
        A = pd.Series(a.squeeze(), name='sensitive_features').astype(int)
        return X, A, Y

    def sample_dataset(self, T, seed):
        x, a, y = self.distribution.sample_train_dataset(T, seed)
        X = pd.Series(x.squeeze(), name='features')
        Y = pd.Series(y.squeeze(), name='label').astype(int)
        A = pd.Series(a.squeeze(), name='sensitive_features').astype(int)

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
        dataset = pd.concat([X, L, A, Y], axis=1)
        return dataset


    def get_loss(self, d, y):

        # get transformed loss vector
        l0 = pd.Series(0.5 * np.ones(len(d)), index=y.index)
        if type(d) is not np.ndarray:
            d = d.to_numpy()
        loss = pd.concat([(1-d)*l0, d*(1-y)], axis=1)
        loss.columns = range(loss.shape[1])
        return loss






