import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

class ClassifierH(object):

    def __init__(self):
        self.error = pd.Series()
        self.gamma = pd.DataFrame()
        self.name = ""

    def set_error(self, error_value):
        self.error = error_value

    def get_error(self):
        return self.error

    def set_gamma(self, gamma_value):
        self.gamma = gamma_value

    def get_gamma(self):
        return self.gamma

    def predict(self, X):
        raise NotImplementedError()


class AcceptAll(ClassifierH):
    def __init__(self):
        super().__init__()
        self.name = 'AccAll'
    def get_name(self):
        return self.name
    def predict(self, X):
        # deleted .tolist()
        pred = pd.Series(np.ones(np.size(X,0), dtype=int)).values
        return pred

class DenyAll(ClassifierH):
    def __init__(self):
        super().__init__()
        self.name = 'DenyAll'
    def get_name(self):
        return self.name
    def predict(self, X):
        return pd.Series(np.zeros(np.size(X,0), dtype=int))

class SensitiveFlip(ClassifierH):
    def __init__(self):
        super().__init__()
        self.name = 'SensFlip'
    def get_name(self):
        return self.name
    def predict(self, X):
        dec = pd.concat([X.loc[:,'sensitive_features_X']==0], axis=0).astype(int)
        #dec = flip.values.tolist()
        return dec

class SensitiveEqual(ClassifierH):
    def __init__(self):
        super().__init__()
        self.name = 'SensEqual'

    def predict(self, XA):
        # deleted .to_list()
        dec = XA.loc[:,'sensitive_features_X']
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

