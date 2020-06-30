# Copyright (c) 2016 akshaykr, adapted by mrateike
import numpy as np
import pandas as pd

"""
adapted wrapper class for the stochastic fair policy retured
by the oracle 
"""

class Policy(object):
    def __init__(self):
        pass
    def get_decision(self, features, L):
        pass
    def get_all_decisions(self, features, K, L):
        pass

class RegressionPolicy(Policy):

    def __init__(self, model):
        self.model = model

    def get_decision(self, x):
        decision = self.model.predict(x)
        return decision

    def predict(self, x):
        return self.model.predict(x)

    def get_classifiers(self):
        classifiers = self.model.pilist
        return classifiers


    def get_all_decisions(self, x):
        # get decisions of policy for a vector of x
        # returns a dataframe with the decisions
        # and the probability with which the decision was taken
        if len(x.index) == 0:
            return []
        dec, prob = self.model.predict(x)
        prob = pd.DataFrame(prob)

        prob_dec = prob.lookup(prob.index, dec).tolist()
        dec_prob = list(zip(dec, prob_dec))
        dec_prob = pd.DataFrame(dec_prob)
        return dec_prob


