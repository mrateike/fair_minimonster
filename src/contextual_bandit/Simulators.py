import pandas as pd
from src.contextual_bandit.Policy import *
from src.contextual_bandit import ContextIterators


class DatasetBandit(object):

    def __init__(self, distribution):
        if distribution == 'Uncalibrated':
            self.distribution = ContextIterators.UncalibratedContextIterator()
        elif distribution == 'FICO':
            self.distribution = ContextIterators.FICODistributionContextIterator()
        elif distribution == 'Adult':
            self.distribution = ContextIterators.AdultCreditDistributionContextIterator()
        elif distribution == 'COMPAS':
            self.distribution = ContextIterators.COMPASDistributionContextIterator()
        else:
            print('SIMULATOR ERROR')

    def get_new_context_set(self, T):
        i = 0
        XA = pd.DataFrame()
        L = pd.DataFrame(columns=['l0', 'l1'])
        A = pd.Series()
        Y = pd.Series()

        while i < T:
            xa, y, a = self.distribution.next()

            XA = XA.append(xa, ignore_index=True).astype(float)

            if y.values == 0:
                L.at[i, 'l0'] = 0
                L.at[i, 'l1'] = 1
            elif y.values == 1:
                L.at[i, 'l0'] = 1
                L.at[i, 'l1'] = 0
            else:
                print('ERROR')

            Y = Y.append(a, ignore_index=True).astype(int)
            A = A.append(a, ignore_index=True).astype(int)
            i += 1

        A = A.rename('sensitive_features')
        Y = Y.rename('label')

        dataset = pd.concat([XA, L, A, Y], axis=1)

        return dataset


    def get_new_context(self):
        return self.distribution.next()

    def get_loss(self, d, y):
        # y is pd.Series
        # d is int

        if d == 0:
            loss = 0.5
        else:
            if y.values == 0:
                # d = 1, y = 0
                loss = 1
            else:
                # d = 1, y = 1
                loss = 0
        return loss




