import pandas as pd
# from data.uncalibrated_score import UncalibratedScore
from data.distribution import UncalibratedScore, FICODistribution, AdultCreditDistribution, COMPASDataset
#from data.uncalibrated_score import UncalibratedScore
"""
This is the main place where we parse different datasets.
For each dataset, implement a ContextIterator with a next method,
that reads the data file and returns a context and reward object.
"""


class ContextIterator(object):

    def __init__(self):
        self.fraction_protected = 0.5
        self.test_percentage = 0.2

    def next(self):
        x, a, y = self.distribution.sample_train_dataset(1)
        X = pd.Series(x.squeeze())
        Y = pd.Series(y.squeeze(), name='label').astype(int)
        A = pd.Series(a.squeeze(), name='sensitive_features_X').astype(int)
        XA = pd.concat([X, A == 1], axis=1).astype(float)
        # XA = pd.concat([X, A], axis=1).astype(float)
        A = A.rename('sensitive_features')
        return XA, Y, A

class UncalibratedContextIterator(ContextIterator):
    def __init__(self):
        super(UncalibratedContextIterator, self).__init__()
        self.distribution = UncalibratedScore(self.fraction_protected)

class FICODistributionContextIterator(ContextIterator):
    def __init__(self):
        super(FICODistributionContextIterator,self).__init__()
        self.distribution = FICODistribution(self.fraction_protected)

class COMPASDistributionContextIterator(ContextIterator):
    def __init__(self):
        super(COMPASDistributionContextIterator, self).__init__()
        self.distribution = COMPASDistribution(self.fraction_protected)

class AdultCreditDistributionContextIterator(ContextIterator):
    def __init__(self):
        super(AdultCreditDistributionContextIterator, self).__init__()
        self.distribution = AdultCreditDistribution(self.test_percentage)




    
