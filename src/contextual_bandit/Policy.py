import numpy as np

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
        return self.model.predict(x)

    def predict(self, x):
        return self.model.predict(x)


    def get_classifiers(self):
        classifiers = self.model.pilist
        return classifiers


    def get_all_decisions(self, x):
        # input: DF

        if len(x.index) == 0:
            return []

        return self.model.predict(x)


