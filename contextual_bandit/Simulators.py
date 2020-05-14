import numpy as np
import pandas as pd
import sys
import sklearn.tree
from contextual_bandit import Argmin
from contextual_bandit.Policy import *
import string
from contextual_bandit import ContextIterators



class DatasetBandit(object):

    def __init__(self, shifts):
        self.distribution = ContextIterators.UncalibratedContextIterator(shifts)

    def get_new_context_set(self, T):
        i = 0
        XA = pd.DataFrame()
        Y = pd.Series()
        A = pd.Series()

        while i < T:
            xa, y, a = self.distribution.next()
            XA = XA.append(xa, ignore_index=True).astype(float)
            Y = Y.append(y, ignore_index=True).astype(int)
            A = A.append(a, ignore_index=True).astype(int)
            i += 1

        Y = Y.rename('label')
        A = A.rename('sensitive_features')

        dataset = pd.concat([XA, Y, A], axis=1)
        return dataset


    def get_new_context(self):
        return self.distribution.next()

    def get_loss(self, d, y):
        # y is pd.Series
        # d is int

        if d == 0:
            loss = [0.5,0.5]
        else:
            if y.values == 0:
                # d = 1, y = 0
                loss = [0,1]
            else:
                # d = 1, y = 1
                loss = [1,0]
        return loss




