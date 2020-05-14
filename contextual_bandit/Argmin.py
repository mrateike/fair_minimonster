from contextual_bandit.Policy import *
import pandas as pd
from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import ExponentiatedGradient
# from fairlearn.reductions._moments.conditional_selection_rate_bechavod import TruePositiveRateDifference
from fairlearn.reductions._moments.conditional_selection_rate import DemographicParity, TruePositiveRateDifference
import numpy as np
import pandas as pd
from IPython.display import display, HTML
_L0 = "l0"
_L1 = "l1"
from sklearn.linear_model import LogisticRegression
# import matplotlib.pyplot as plt
from scipy.stats import cumfreq
import random
from data.uncalibrated_score import UncalibratedScore


def argmin(eps, nu, fairness, dataset1, dataset2=None):


    if dataset2 is None:
        # _X1 = dataset1.drop(['sensitive_features', 'label'], axis=1)
        _y = dataset1.loc[:, 'label']
        # _sensitive_features1 = dataset1.loc[:, 'sensitive_features']
        XA = pd.DataFrame(dataset1.iloc[:, 0:2])
        A = pd.Series(dataset1.loc[:, 'sensitive_features'], name='sensitive_features')
        Y = pd.Series(dataset1.loc[:, 'label'], name = 'label')
        L = pd.DataFrame(columns=['l0', 'l1'])

        for index, value in _y.items():
            if value == 0:
                L.at[index, _L0] = 0
                L.at[index, _L1] = 1
            elif value == 1:
                L.at[index, _L0] = 1
                L.at[index, _L1] = 0
            else:
                print('ERROR')
    else:
        A = dataset2.loc[:, 'sensitive_features']
        L = dataset2.filter(items=['l0', 'l1'])
        XA = dataset2.drop(columns=['sensitive_features', 'l0', 'l1'])




    expgrad_XA = ExponentiatedGradient(
        dataset1,
        LogisticRegression(solver='liblinear', fit_intercept=True),
        constraints=fairness,
        eps=eps,
        nu=nu)

    expgrad_XA.fit(
        XA,
        L,
        sensitive_features=A)

    _y = dataset1.loc[:, 'label']
    # _sensitive_features1 = dataset1.loc[:, 'sensitive_features']
    XA = pd.DataFrame(dataset1.iloc[:, 0:2])
    Y = pd.Series(dataset1.loc[:, 'label'], name='label')
    logReg_predictor = LogisticRegression(solver='liblinear', fit_intercept=True)
    logReg_predictor.fit(XA, Y)

    #
    #
    # # print('--- validation oracle policy returned --- ')
    # fraction_protected = 0.5
    # n_test = 5000
    # # shifts == True (DP), shifts = False (EO, TRP)
    # shifts = False
    # distribution = UncalibratedScore(fraction_protected)
    # x_test, a_test, y_test = distribution.sample_test_dataset(n_test, shifts)
    # X_test = pd.DataFrame(x_test.squeeze())
    # y_test = pd.Series(y_test.squeeze(), name='label')
    # A_test = pd.Series(a_test.squeeze(), name='sensitive_features_X')
    # XA_test = pd.concat([X_test, A_test == 1], axis=1).astype(float)
    # A_test = A_test.rename('sensitive_features')
    #
    # def summary_as_df(name, summary):
    #     a = pd.Series(summary.by_group)
    #     a['overall'] = summary.overall
    #     return pd.DataFrame({name: a})
    #
    # scores_expgrad_XA = pd.Series(expgrad_XA.predict(XA_test), name="scores_expgrad_XA")
    #
    # auc_expGrad = summary_as_df(
    #     "accuracy_XA",
    #     accuracy_score_group_summary(y_test, scores_expgrad_XA, sensitive_features=A_test))
    # mean_pred_expGrad = summary_as_df(
    #     "acceptance_rate_XA",
    #     mean_prediction_group_summary(y_test, scores_expgrad_XA, sensitive_features=A_test))
    #
    # parity_expGrad = demographic_parity_difference(y_test, scores_expgrad_XA, sensitive_features=A_test)
    # ratio_parity_expGrad = demographic_parity_ratio(y_test, scores_expgrad_XA, sensitive_features=A_test)
    # TPR_expGrad = true_positive_rate_difference(y_test, scores_expgrad_XA, sensitive_features=A_test)
    # ratio_TPR_expGrad = true_positive_rate_ratio(y_test, scores_expgrad_XA, sensitive_features=A_test)
    # FPR_expGrad = false_positive_rate_difference(y_test, scores_expgrad_XA, sensitive_features=A_test)
    # ratio_FPR_expGrad = false_positive_rate_ratio(y_test, scores_expgrad_XA, sensitive_features=A_test)
    # # equalOdds_expGrad = equalized_odds_difference(y_test, scores_expgrad_XA, sensitive_features=A_test)
    # # ratio_equalOdds_expGrad = equalized_odds_ratio(y_test, scores_expgrad_XA, sensitive_features=A_test)
    #
    #
    #
    # classifier_summary = pd.concat([auc_expGrad, mean_pred_expGrad], axis=1)
    # display(classifier_summary)
    #
    # print("DP = ", parity_expGrad)
    # print("DP_ratio = ", ratio_parity_expGrad)
    # print("TPR = ", TPR_expGrad)
    # print("TPR_ratio = ", ratio_TPR_expGrad)
    # print("FPR = ", FPR_expGrad)
    # print("FPR_ratio = ", ratio_FPR_expGrad)
    # # print("EO = ", equalOdds_expGrad)
    # # print("EO_ratio = ", ratio_equalOdds_expGrad)


    return RegressionPolicy(expgrad_XA), RegressionPolicy(logReg_predictor)


