# Copyright (c) 2016 akshaykr, adapted 2020 by mrateike
# Licensed under the MIT License.
import pandas as pd
from data.distribution import UncalibratedScore, FICODistribution

"""
adapted for parsing different datasets.
For each dataset, implement a ContextIterator, which initializes the 
distributions.
"""


class ContextIterator(object):
    def __init__(self):
        # distribution between two sensitive groups
        self.fraction_protected = 0.5

class UncalibratedContextIterator(ContextIterator):
    def __init__(self):
        super(UncalibratedContextIterator, self).__init__()
        self.distribution = UncalibratedScore(self.fraction_protected)

class FICODistributionContextIterator(ContextIterator):
    def __init__(self):
        super(FICODistributionContextIterator,self).__init__()
        self.distribution = FICODistribution(self.fraction_protected)





    
