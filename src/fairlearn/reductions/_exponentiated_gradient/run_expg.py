import os
import sys

root_path = os.path.abspath(os.path.join(".."))
if root_path not in sys.path:
    sys.path.append(root_path)
from .._exponentiated_gradient.exponentiated_gradient  import ExponentiatedGradient
import numpy as np
import pandas as pd
import matplotlib as mpl
#TkAgg
mpl.use('Agg')
import matplotlib.pyplot as plt


from sklearn.linear_model import LogisticRegression
from src.fairlearn.metrics import mean_prediction_group_summary, accuracy_score_group_summary, equalized_odds_difference, demographic_parity_difference, demographic_parity_ratio, equalized_odds_ratio, true_positive_rate_difference, true_positive_rate_ratio, false_positive_rate_difference, false_positive_rate_ratio
# import matplotlib.pyplot as plt
from data.uncalibrated_score import UncalibratedScore

import random

# Load the data using the tempeh package
# from tempeh.configurations import datasets
# dataset = datasets['lawschool_passbar']()
#
# X_train, X_test = dataset.get_X(format=pd.DataFrame)
# y_train, y_test = dataset.get_y(format=pd.Series)
# A_train, A_test = dataset.get_sensitive_features(name='race', format=pd.Series)

def play (n1_train, n2_train, fairness, eps, nu, statistics, seed):

    fraction_protected = 0.5
    distribution = UncalibratedScore(fraction_protected)
    x1_train, a1_train, y1_train = distribution.sample_train_dataset(n1_train, seed[0])
    x2_train, a2_train, y2_train = distribution.sample_train_dataset(n2_train, seed[1])

    y2_train = pd.Series((y2_train.squeeze()))
    l_hat = pd.DataFrame(np.ones((x2_train.shape[0], 2)), index=range(len(y2_train)), columns=['l0', 'l1'])

    index = 0

    for y_true in y2_train.tolist():
        # # ---- supervised learning -----
        # if y_true == 0:
        #     l_hat.at[index, 'l0'] = 0
        #     l_hat.at[index, 'l1'] = 1
        # else:
        #     l_hat.at[index, 'l0'] = 1
        #     l_hat.at[index, 'l1'] = 0
        #index += 1

        # -----  bandit min Loss -------
        # right = int(random.uniform(0,1)>0.3)
        # if right == 0:
        #     if y_true == 0:
        #         #dec == 1
        #         l_hat.at[index, 'l0'] = 0/0.7
        #         l_hat.at[index, 'l1'] = 1/0.3
        #     else:
        #         # dec == 0
        #         l_hat.at[index, 'l0'] = 0.5/0.3
        #         l_hat.at[index, 'l1'] = 0.5/0.7
        # else:
        #     if y_true == 0:
        #         # dec == 0
        #         l_hat.at[index, 'l0'] = 0.5/0.7
        #         l_hat.at[index, 'l1'] = 0.5/0.3
        #     else:
        #         # dec == 1
        #         l_hat.at[index, 'l0'] = 1/0.3
        #         l_hat.at[index, 'l1'] = 0/0.7
        # index += 1

        # -----  bandit min Loss version 2 -------
        p = 0.1
        right = int(random.uniform(0,1)>p)
        if right == 0:
            if y_true == 0:
                #dec == 1
                l_hat.at[index, 'l0'] = 0
                l_hat.at[index, 'l1'] = 1/p
            else:
                # dec == 0
                l_hat.at[index, 'l0'] = 0.5/p
                l_hat.at[index, 'l1'] = 0
        else:
            if y_true == 0:
                # dec == 0
                l_hat.at[index, 'l0'] = 0.5/(1-p)
                l_hat.at[index, 'l1'] = 0
            else:
                # dec == 1
                l_hat.at[index, 'l0'] = 0
                l_hat.at[index, 'l1'] = 0/(1-p)
        index += 1

        # -----  bandit max Var -------
        # right = int(random.uniform(0, 1) > 0.3)
        # mu =0
        # if right == 0:
        #     if y_true == 0:
        #         # dec == 1
        #         l_hat.at[index, 'l0'] = (0-mu) / 0.7
        #         # l_hat.at[index, 'l1'] = (1-mu) / 0.3
        #     else:
        #         # dec == 0
        #         l_hat.at[index, 'l0'] = (0.5-mu) / 0.3
        #         l_hat.at[index, 'l1'] = (0.5-mu) / 0.7
        # else:
        #     if y_true == 0:
        #         # dec == 0
        #         l_hat.at[index, 'l0'] = (0.5-mu) / 0.7
        #         l_hat.at[index, 'l1'] = (0.5-mu) / 0.3
        #     else:
        #         # dec == 1
        #         l_hat.at[index, 'l0'] = (1-mu) / 0.3
        #         l_hat.at[index, 'l1'] = (0-mu) / 0.7
        # index += 1

        # #-----  bandit max Var version 2 -------
        # right = int(random.uniform(0, 1) > 0.3)
        # mu = 0.1
        # if right == 0:
        #     if y_true == 0:
        #         # dec == 1
        #         l_hat.at[index, 'l0'] = (1) - mu / 0.7
        #         l_hat.at[index, 'l1'] = (1-mu) / 0.3
        #     else:
        #         # dec == 0
        #         l_hat.at[index, 'l0'] = (0.5-mu) / 0.3
        #         l_hat.at[index, 'l1'] = (0.5)-mu / 0.7
        # else:
        #     if y_true == 0:
        #         # dec == 0
        #         l_hat.at[index, 'l0'] = (0.5-mu) / 0.7
        #         l_hat.at[index, 'l1'] = (0.5)-mu / 0.3
        #     else:
        #         # dec == 1
        #         l_hat.at[index, 'l0'] = (1)-mu / 0.3
        #         l_hat.at[index, 'l1'] = (1-mu) / 0.7
        # index += 1


    l_IPS  = l_hat.set_index(y2_train.index)

    X1_train = pd.DataFrame(x1_train.squeeze())
    X2_train = pd.DataFrame(x2_train.squeeze())
    # X_test = pd.DataFrame(x_test.squeeze())

    y1_train = pd.Series(y1_train.squeeze(), name='label')
    # y2_train = pd.Series(y2_train.squeeze(), name='label')
    # y_test = pd.Series(y_test.squeeze(), name='label')

    # need a different name for classifier family sens_equal and sens_flip
    A1_train = pd.Series(a1_train.squeeze(), name='sensitive_features_X')
    A2_train = pd.Series(a2_train.squeeze(), name='sensitive_features_X')
    # A_test = pd.Series(a_test.squeeze(), name='sensitive_features_X')

    # black, women == 1
    XA1_train = pd.concat([X1_train, A1_train==1], axis=1).astype(float)
    XA2_train = pd.concat([X2_train, A2_train==1], axis=1).astype(float)
    # XA_test = pd.concat([X_test, A_test==1], axis=1).astype(float)

    A1_train = A1_train.rename('sensitive_features')
    A2_train = A2_train.rename('sensitive_features')
    # A_test = A_test.rename('sensitive_features')

    L1_train = pd.DataFrame(columns=['l0', 'l1'])
    for i, values in y1_train.items():
        if values == 0:
            L1_train.at[i, 'l0'] = 0
            L1_train.at[i, 'l1'] = 1
        elif values == 1:
            L1_train.at[i, 'l0'] = 1
            L1_train.at[i, 'l1'] = 0
        else:
            print('ERROR')

    # Combine all training data into a single data frame and glance at a few rows
    dataset_phase1XA = pd.concat([XA1_train, L1_train, A1_train, y1_train], axis=1)


    expgrad_XA = ExponentiatedGradient(
        dataset_phase1XA,
        LogisticRegression(solver='liblinear', fit_intercept=True),
        constraints=fairness,
        eps=eps,
        nu=nu)

    expgrad_XA.fit(
        XA2_train,
        l_IPS,
        sensitive_features=A2_train)

    statistics.evaluate(expgrad_XA)

    accuracy = statistics.acc_list_overall
    accuracy1 = statistics.acc_list_1
    accuracy0 = statistics.acc_list_0
    mean_pred = statistics.mean_pred_overall_list
    mean_pred1 = statistics.mean_pred_0_list
    mean_pred0 = statistics.mean_pred_1_list
    util = statistics.util_list
    DP = statistics.DP_list
    FPR = statistics.FPR_list
    scores = statistics.scores_dict

    acc_dict = {0: accuracy1, 1: accuracy0,
                'overall': accuracy}
    pred_dict = {0: mean_pred0, 1: mean_pred1,
                 'overall': mean_pred}

    results_dict = {'acc_dict': acc_dict, 'pred_dict':pred_dict, 'util': util, 'DP':DP, 'FPR':FPR}

    return results_dict, scores


    # #---- Evaluation of log regressor -----
    # logReg_predictor = LogisticRegression(solver='liblinear', fit_intercept=True)
    # logReg_predictor.fit(XA2_train, y2_train)
    #
    # # a convenience function that transforms the result of a group metric call into a data frame
    # def summary_as_df(name, summary):
    #     a = pd.Series(summary.by_group)
    #     a['overall'] = summary.overall
    #     return pd.DataFrame({name: a})
    #
    # scores_logReg = pd.Series(logReg_predictor.predict(XA_test), name="score_unmitigated")
    #
    # auc_logReg = summary_as_df(
    #      "accuracy_unmitigated_predictor",
    #      accuracy_score_group_summary(y_test, scores_logReg, sensitive_features=A_test))
    # sel_logReg = summary_as_df(
    #     "selection_unmitigated_predictor",
    #     mean_prediction_group_summary(y_test, scores_logReg, sensitive_features=A_test))
    #
    # classifier_summary = pd.concat([auc_logReg, sel_logReg], axis=1)
    #
    # parity_logReg = demographic_parity_difference(y_test, scores_logReg, sensitive_features=A_test)
    # ratio_parity_logReg = demographic_parity_ratio(y_test, scores_logReg, sensitive_features=A_test)
    # TPR_logReg = true_positive_rate_difference(y_test, scores_logReg, sensitive_features=A_test)
    # ratio_TPR_logReg = true_positive_rate_ratio(y_test, scores_logReg, sensitive_features=A_test)
    # FPR_logReg = false_positive_rate_difference(y_test, scores_logReg, sensitive_features=A_test)
    # ratio_FPR_logReg = false_positive_rate_ratio(y_test, scores_logReg, sensitive_features=A_test)
    # equalOdds_logReg = equalized_odds_difference(y_test, scores_logReg, sensitive_features=A_test)
    # ratio_equalOdds_logReg = equalized_odds_ratio(y_test, scores_logReg, sensitive_features=A_test)
    #
    # print("----- LogReg ---------------")
    # print("DP = ", parity_logReg)
    # print("DP_ratio = ", ratio_parity_logReg)
    # print("TPR = ", TPR_logReg)
    # print("TPR_ratio = ", ratio_TPR_logReg)
    # print("FPR = ", FPR_logReg)
    # print("FPR_ratio = ", ratio_FPR_logReg)
    # print("EO = ", equalOdds_logReg)
    # print("EO_ratio = ", ratio_equalOdds_logReg)

    #
    #
    # return accuracy, mean_pred, util, DP, TPR, EO






    #plt.savefig(base_save_path + '/plot.png')
    # scores_expgrad_XA = pd.Series(expgrad_XA.predict(XA_test), name="scores_expgrad_XA")
    # auc_expGrad = summary_as_df(
    #     "accuracy_XA",
    #     accuracy_score_group_summary(y_test, scores_expgrad_XA, sensitive_features=A_test))
    # mean_pred_expGrad = summary_as_df(
    #     "acceptance_rate_XA",
    #     mean_prediction_group_summary(y_test, scores_expgrad_XA, sensitive_features=A_test))


    # parity_expGrad = demographic_parity_difference(y_test, scores_expgrad_XA, sensitive_features=A_test)
    # ratio_parity_expGrad = demographic_parity_ratio(y_test, scores_expgrad_XA, sensitive_features=A_test)
    # TPR_expGrad = true_positive_rate_difference(y_test, scores_expgrad_XA, sensitive_features=A_test)
    # ratio_TPR_expGrad = true_positive_rate_ratio(y_test, scores_expgrad_XA, sensitive_features=A_test)
    # FPR_expGrad = false_positive_rate_difference(y_test, scores_expgrad_XA, sensitive_features=A_test)
    # ratio_FPR_expGrad = false_positive_rate_ratio(y_test, scores_expgrad_XA, sensitive_features=A_test)
    # equalOdds_expGrad = equalized_odds_difference(y_test, scores_expgrad_XA, sensitive_features=A_test)
    # ratio_equalOdds_expGrad = equalized_odds_ratio(y_test, scores_expgrad_XA, sensitive_features=A_test)

    # classifier_summary = pd.concat([auc_expGrad, mean_pred_expGrad], axis=1)
    # # classifier_summary.loc['disparity']=(classifier_summary.iloc[1]-classifier_summary.iloc[0]).abs()
    # # classifier_summary.loc['disparity', classifier_summary.columns.str.startswith('auc')]='-'
    # # display(classifier_summary)
    #
    # print("----- ExpGrad ---------------")
    # print("DP = ", parity_expGrad)
    # print("DP_ratio = ", ratio_parity_expGrad)
    # print("TPR = ", TPR_expGrad)
    # print("TPR_ratio = ", ratio_TPR_expGrad)
    # print("FPR = ", FPR_expGrad)
    # print("FPR_ratio = ", ratio_FPR_expGrad)
    # print("EO = ", equalOdds_expGrad)
    # print("EO_ratio = ", ratio_equalOdds_expGrad)


    # fig_train = plt.figure()
    # # It's the arrangement of subgraphs within this graph. The first number is how many rows of subplots; the second number is how many columns of subplots; the third number is the subgraph you're talking about now. In this case, there's one row and one column of subgraphs (i.e. one subgraph) and the axes are talking about the first of them. Something like fig.add_subplot(3,2,5) would be the lower-left subplot in a grid of three rows and two columns
    # ax1_train = fig_train.add_subplot(111)
    # ax1_train.scatter(range(0, len(accuracy)), accuracy, label='accuracy')
    # ax1_train.scatter(range(0, len(DP)), DP, label='DP')
    # plt.xlabel("iterations")
    # plt.ylabel("DP/accuracy")
    # plt.title('Full Bandit Run')
    # plt.legend()
    # #plt.savefig(base_save_path + '/plot.png')
    #
    # improved_DP = parity_logReg - parity_expGrad
    # improved_ratio_parity = ratio_parity_expGrad - ratio_parity_logReg
    # improved_TPR = TPR_logReg - TPR_expGrad
    # improved_ratio_TPR = ratio_TPR_expGrad - ratio_TPR_logReg
    # improved_FPR = FPR_logReg - FPR_expGrad
    # improved_ratio_FPR = ratio_FPR_expGrad - ratio_FPR_logReg
    # improved_equalOdds = equalOdds_logReg - equalOdds_expGrad
    # improved_ratio_equalOdds = ratio_equalOdds_expGrad - ratio_equalOdds_logReg
    # improved_acc = (auc_expGrad.loc['overall'].values - auc_logReg.loc['overall'].values).squeeze()
    #
    # print("------- IMPROVEMENT of Fairlearn --------------")
    # print("DP = ", improved_DP)
    # print("DP_ratio = ", improved_ratio_parity)
    # print("TPR = ", improved_TPR)
    # print("TPR_ratio = ", improved_ratio_TPR)
    # print("FPR = ", improved_FPR)
    # print("FPR_ratio = ", improved_ratio_FPR)
    # print("EO = ", improved_equalOdds)
    # print("EO_ratio = ", improved_ratio_equalOdds)
    # print("------- Loss of Fairlearn --------------")
    # print("Accuracy = ", improved_acc)

    # return improved_acc, improved_DP, improved_TPR, improved_FPR, improved_equalOdds