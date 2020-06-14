# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import numpy as np
import pandas as pd
import statistics
from .moment import ClassificationMoment
from .moment import _ALL

_GROUP_ID = "group_id"
_L0 = "l0"
_L1 = "l1"
class ErrorRate(ClassificationMoment):
    """Misclassification error."""

    #short_name = "Err"

    def load_data(self, X, loss):
        """Load the specified data into the object."""

        # super().load_data(X, y, **kwargs)

        self.X = X
        self.tags = pd.DataFrame(loss)

        self.index = [_ALL]

        self.X_all = pd.concat([self.X_all , self.X], axis = 0, ignore_index=True)
        self.tags_all = pd.concat([self.tags_all, self.tags], axis = 0, ignore_index=True)



    def load_data1(self, X, loss):

        self.X_all = X
        self.tags_all = pd.DataFrame(loss)
        self.index = [_ALL]




    def gamma(self, predictor):
        """Return the gamma values for the given predictor. predictor is always a classifier h"""
        # evaluated on both datasets
        pred = predictor(self.X_all)

        error = [0]
        index = 0
        for value in pred:
            if value == 0:
                error.append(self.tags_all.loc[index, _L0])
            else:
                error.append(self.tags_all.loc[index, _L1])
            index += 1

        error = statistics.mean(error[1:])

        return error

    def project_lambda(self, lambda_vec):
        """Return the lambda values."""
        return lambda_vec

    def signed_weights(self):
        """Return the signed weights."""
        return self.tags_all[_L0] - self.tags_all[_L1]