from src.contextual_bandit.Policy import *
from src.fairlearn.reductions._exponentiated_gradient.exponentiated_gradient import ExponentiatedGradient
# from fairlearn.reductions._moments.conditional_selection_rate_bechavod import TruePositiveRateDifference
import pandas as pd
from data.util import save_dictionary
from src.evaluation.Evaluation import my_plot
from src.evaluation.Evaluation import Evaluation, my_plot
from src.evaluation.training_evaluation import Statistics
from data.util import save_dictionary
from src.evaluation.training_evaluation import UTILITY

_L0 = "l0"
_L1 = "l1"
from sklearn.linear_model import LogisticRegression
# import matplotlib.pyplot as plt


def argmin(randthresh, eps, nu, fairness, dataset1, dataset2=None):

    expgrad_XA = ExponentiatedGradient(randthresh,
        dataset1,
        LogisticRegression(solver='liblinear', fit_intercept=True),
        constraints=fairness,
        eps=eps,
        nu=nu)

    if dataset2.empty:
        expgrad_XA.fit()

    else:
        A = dataset2.loc[:, 'sensitive_features']
        L = dataset2.loc[:,['l0', 'l1']]
        XA = dataset2.loc[:,['features', 'sensitive_features']]
        # print('A', A)
        # print('L', L)
        # print('XA', XA)
        expgrad_XA.fit(
            XA,
            L,
            sensitive_features=A)

    return RegressionPolicy(expgrad_XA)


