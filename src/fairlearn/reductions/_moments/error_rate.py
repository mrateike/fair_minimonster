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

    short_name = "Err"

    def load_data(self, X, y, **kwargs):
        """Load the specified data into the object."""

        # super().load_data(X, y, **kwargs)

        self.X = X
        self.tags = pd.DataFrame(y)
        self.index = [_ALL]

        self.X_all = pd.concat([self.X1, self.X], axis = 0, ignore_index=True)
        self.tags_all = pd.concat([self.tags1, self.tags], axis = 0, ignore_index=True)


    def load_data1(self, X, y, **kwargs):

        # super().load_data1(X, y, **kwargs)

        self.X1 = X
        n = self.X1.shape[0]
        data = np.zeros((n,2))
        self.tags1 = pd.DataFrame(data, columns=[_L0,_L1])
        self.index1 = [_ALL]

        index = 0
        for value in y:
            if value == 0 :
                self.tags1.at[index,_L0] =0
                self.tags1.at[index,_L1] =1
            elif value == 1:
                self.tags1.at[index, _L0] =1
                self.tags1.at[index, _L1] =0
            else:
                print('ERROR')
            index +=1


    def gamma(self, predictor):
        """Return the gamma values for the given predictor."""
        # evaluated on both datasets
        pred = predictor(self.X_all)

        # error = pd.Series(data=(self.tags_all[_LABEL] - pred).abs().mean(),
        #                   index=self.index)

        error = [0]
        index = 0
        for value in pred:
            if value == 0:
                errortoappend = self.tags_all.loc[index, _L0]
                error.append(errortoappend)
            else:
                errortoappend = self.tags_all.loc[index, _L1]
                error.append(errortoappend)
            index += 1
        error = error[1:]
        error = statistics.mean(error)
        self._gamma_descr = str(error)
        return error

    def project_lambda(self, lambda_vec):
        """Return the lambda values."""
        return lambda_vec

    def signed_weights(self, lambda_vec=None):
        """Return the signed weights."""
        return self.tags_all[_L0] - self.tags_all[_L1]