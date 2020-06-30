# Copyright (c) 2020 mrateike

import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

"""
Implementing the classifier family containing four constant classifiers by assumption of
Bechavod, Yahav, et al. "Equal opportunity in online classification with partial feedback." 
Advances in Neural Information Processing Systems. 2019.
"""

class ClassifierH(object):

    def __init__(self):
        self.name = ""

    def get_name(self):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()


class AcceptAll(ClassifierH):
    # classifier deciding constantly d = 1
    def __init__(self):
        super().__init__()
        self.name = 'AccAll'
    def get_name(self):
        return self.name
    def predict(self, X):
        dec = pd.Series(np.ones(np.size(X,0), dtype=int)).values
        return dec

class DenyAll(ClassifierH):
    # classifier deciding constantly d = 0
    def __init__(self):
        super().__init__()
        self.name = 'DenyAll'
    def get_name(self):
        return self.name
    def predict(self, X):
        dec  = pd.Series(np.zeros(np.size(X,0), dtype=int)).values
        return dec

class SensitiveFlip(ClassifierH):
    # classifier identity function of a = 0
    def __init__(self):
        super().__init__()
        self.name = 'SensFlip'
    def get_name(self):
        return self.name
    def predict(self, X):
        dec = pd.concat([X.loc[:,'sensitive_features']==0], axis=0).astype(int).values
        return dec

class SensitiveEqual(ClassifierH):
    # classifier identity function of a = 1
    def __init__(self):
        super().__init__()
        self.name = 'SensEqual'

    def predict(self, XA):
        dec = XA.loc[:,'sensitive_features'].values
        return dec

class ClassifierFamily(object):

    def __init__(self):
        aa = AcceptAll()
        def h_aa(X): return aa.predict(X)
        da = DenyAll()
        def h_da(X): return da.predict(X)
        se = SensitiveEqual()
        def h_se(X): return se.predict(X)
        sf = SensitiveFlip()
        def h_sf(X): return sf.predict(X)
        self.classifiers = pd.Series([h_aa, h_da, h_se, h_sf])

