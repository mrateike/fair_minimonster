from src.contextual_bandit.Policy import *
from src.fairlearn.reductions._exponentiated_gradient.exponentiated_gradient import ExponentiatedGradient
# from fairlearn.reductions._moments.conditional_selection_rate_bechavod import TruePositiveRateDifference
import pandas as pd

_L0 = "l0"
_L1 = "l1"
from sklearn.linear_model import LogisticRegression
# import matplotlib.pyplot as plt


def argmin(eps, nu, fairness, dataset1, dataset2=None):


    expgrad_XA = ExponentiatedGradient(
        dataset1,
        LogisticRegression(solver='liblinear', fit_intercept=True),
        constraints=fairness,
        eps=eps,
        nu=nu)

    if dataset2 is None:
        expgrad_XA.fit()

        # _y = dataset1.loc[:, 'label']
        # XA = dataset1.drop(columns=['sensitive_features', 'label'])
        # A = pd.Series(dataset1.loc[:, 'sensitive_features'], name='sensitive_features')
        # L = pd.DataFrame(columns=['l0', 'l1'])
        #
        # for index, value in _y.items():
        #     if value == 0:
        #         L.at[index, _L0] = 0
        #         L.at[index, _L1] = 1
        #     elif value == 1:
        #         L.at[index, _L0] = 1
        #         L.at[index, _L1] = 0
        #     else:
        #         print('ERROR')
    else:
        A = dataset2.loc[:, 'sensitive_features']
        L = dataset2.filter(items=['l0', 'l1'])
        XA = dataset2.drop(columns=['sensitive_features', 'l0', 'l1'])

        expgrad_XA.fit(
            XA,
            L,
            sensitive_features=A)


    # XA = dataset1.drop(columns=['sensitive_features', 'label'])
    # Y = pd.Series(dataset1.loc[:, 'label'], name='label')
    #
    # logReg_predictor = LogisticRegression(solver='liblinear', fit_intercept=True)
    # logReg_predictor.fit(XA, Y)

    return RegressionPolicy(expgrad_XA)


