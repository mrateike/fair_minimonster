from src.contextual_bandit.Policy import *
from src.fairlearn.reductions._exponentiated_gradient.exponentiated_gradient import ExponentiatedGradient
# from fairlearn.reductions._moments.conditional_selection_rate_bechavod import TruePositiveRateDifference
import pandas as pd
from data.util import save_dictionary
from src.evaluation.Evaluation import my_plot
from src.evaluation.Evaluation import Evaluation, save_and_plot_results, my_plot
from src.evaluation.training_evaluation import Statistics
from data.util import save_dictionary
from src.evaluation.training_evaluation import UTILITY

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

    else:
        A = dataset2.loc[:, 'sensitive_features']
        L = dataset2.filter(items=['l0', 'l1'])
        XA = dataset2.drop(columns=['sensitive_features', 'l0', 'l1'])

        expgrad_XA.fit(
            XA,
            L,
            sensitive_features=A)

    return RegressionPolicy(expgrad_XA)


