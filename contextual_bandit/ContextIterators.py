import pandas as pd
from data.uncalibrated_score import UncalibratedScore

"""
This is the main place where we parse different datasets.
For each dataset, implement a ContextIterator with a next method,
that reads the data file and returns a context and reward object.
"""

# author: Miriam - not finished
class UncalibratedContextIterator(object):
    def __init__(self, shifts):
        self.shifts = shifts
        self.curr_idx = 0
        fraction_protected = 0.5
        self.distribution = UncalibratedScore(fraction_protected)

    def next(self):
        x, a, y = self.distribution.sample_train_dataset(1,self.shifts)
        X = pd.Series(x.squeeze())
        Y = pd.Series(y.squeeze(), name='label').astype(int)
        A = pd.Series(a.squeeze(), name='sensitive_features_X').astype(int)
        XA = pd.concat([X, A == 1], axis=1).astype(float)
        A = A.rename('sensitive_features')
        return XA, Y, A



    
