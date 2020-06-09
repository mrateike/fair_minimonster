# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""A variety of extra metrics useful for assessing fairness.

These are metrics which are not part of `scikit-learn`.
"""

import sklearn.metrics as skm
import pandas as pd
from ._balanced_root_mean_squared_error import balanced_root_mean_squared_error  # noqa: F401
from ._mean_predictions import mean_prediction, _mean_overprediction, _mean_underprediction  # noqa: F401,E501
from ._selection_rate import selection_rate  # noqa: F401,E501


def true_positive_rate(y_true, y_pred, sample_weight=None):
    r"""Calculate the true positive rate (also called sensitivity, recall, or hit rate)."""
    tnr, fpr, fnr, tpr = skm.confusion_matrix(
        y_true, y_pred, sample_weight=sample_weight, labels=[0, 1], normalize="true").ravel()
    return tpr


def true_negative_rate(y_true, y_pred, sample_weight=None):
    r"""Calculate the true negative rate (also called specificity or selectivity)."""
    tnr, fpr, fnr, tpr = skm.confusion_matrix(
        y_true, y_pred, sample_weight=sample_weight, labels=[0, 1], normalize="true").ravel()
    return tnr


def false_positive_rate(y_true, y_pred, sample_weight=None):
    r"""Calculate the false positive rate (also called fall-out)."""
    tnr, fpr, fnr, tpr = skm.confusion_matrix(
        y_true, y_pred, sample_weight=sample_weight, labels=[0, 1], normalize="true").ravel()
    return fpr


def false_negative_rate(y_true, y_pred, sample_weight=None):
    r"""Calculate the false negative rate (also called miss rate)."""
    tnr, fpr, fnr, tpr = skm.confusion_matrix(
        y_true, y_pred, sample_weight=sample_weight, labels=[0, 1], normalize="true").ravel()
    return fnr


def root_mean_squared_error(y_true, y_pred, **kwargs):
    r"""Calculate the root mean squared error."""
    return skm.mean_squared_error(y_true, y_pred, squared=False, **kwargs)

def test_statistics(y_test, a_test):
        a_true = pd.Series(a_test.squeeze())
        y_true = pd.Series(y_test.squeeze())
        a_true_list = a_true.values.tolist()
        y_true_list = y_true.values.tolist()

        ny1 = 0
        ny0 = 0
        na0 = 0
        na1 = 0
        ny0a0 = 0
        ny0a1 = 0
        ny1a0 = 0
        ny1a1 = 0

        best_pred = 0
        best_preda0 = 0
        best_preda1 = 0

        for i in range(len(y_true_list)):
            best_pred += y_true_list[i]

            if y_true_list[i] == 1:
                ny1 += 1
                if a_true_list[i] == 1:
                    ny1a1 += 1
                else:
                    ny1a0 += 1
            else:
                ny0 += 1
                if a_true_list[i] == 1:
                    ny0a1 += 1
                else:
                    ny0a0 += 1

            if a_true_list[i] == 1:
                na1 += 1
                best_preda1 += y_true_list[i]

            else:
                na0 += 1
                best_preda0 += y_true_list[i]

        n = len(y_true_list)

        pa0 = na0 / n
        pa1 = na1 / n
        py1 = ny1 / n
        py0 = ny0 / n
        py1a0 = ny1a0 / n
        py1a1 = ny1a1 / n
        py0a0 = ny0a0 / n
        py0a1 = ny0a1 / n

        print('--- Probabilities --- ')
        print('n', n)
        print('a0', pa0, 'a1', pa1)
        print('y0', py0, 'y1', py1)
        print('y0a0', py0a0, 'y0a1', py0a1)
        print('y1a0', py1a0, 'y1a1', py1a1)


def utility(y_true, y_pred, sample_weight=None):
    # pred_mean = y_pred.mean(axis=0)
    # # print('pred_mean', pred_mean)
    # proby1 = len(y_true[y_true.values == 1]) / len(y_true)
    # # print('proby1', proby1)
    c = 0.5
    # print('y_true', y_true, type(y_true))
    # print('y_pred', y_pred, type(y_pred))

    #utility = y_pred(y_true - c)
    print('utility: y_true', )
    utility = y_pred.multiply(y_true) - y_pred.multiply(c)
    return utility.mean()

def accuracy2_score(y_true, y_pred, sample_weight=None):
    sum = 0
    for index, yt in y_true.items():
        if yt == y_pred[index]:
            sum +=1
    return sum/len(y_true)



