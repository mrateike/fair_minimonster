import numpy as np
import pandas as pd

class Policy(object):
    """
    A policy object prescribes actions for contexts
    The default policy class prescribes random actions
    """
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

        if len(x.index) == 0:
            return []
        dec, prob = self.model.predict(x)
        prob = pd.DataFrame(prob)

        prob_dec = prob.lookup(prob.index, dec).tolist()
        dec_prob = list(zip(dec, prob_dec))
        dec_prob = pd.DataFrame(dec_prob)
        return dec_prob


