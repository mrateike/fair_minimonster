import numpy as np

"""
Interface for accessing contexts, because different datasets have different looking formats. 
This handles both label-dependent features and non-label dependent features. 
"""
class Context(object):
    def __init__(self, name, features):
        self.name = name
        self.features = features
        self.dim = self.features.shape[1]

    def get_features(self):
        return self.features

    def get_name(self):
        return self.name

    def get_dim(self):
        return self.dim
