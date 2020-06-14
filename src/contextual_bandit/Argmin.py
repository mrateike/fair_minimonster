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


def argmin(base_save_path, statistics, eps, nu, fairness, dataset1, dataset2=None):

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

    statistics.evaluate(expgrad_XA)


    # ------ my extra stats -----
    accuracy = statistics.acc_list_overall
    accuracy1 = statistics.acc_list_1
    accuracy0 = statistics.acc_list_0
    mean_pred = statistics.mean_pred_overall_list
    mean_pred1 = statistics.mean_pred_0_list
    mean_pred0 = statistics.mean_pred_1_list
    util = statistics.util_list
    DP = statistics.DP_list

    # Todo: here we say which EOP to use (TPR, FPR, EOP)
    EOP = statistics.TPR_list
    # FPR = statistics.FPR_list
    # TPR = statistics.TPR_list
    scores = statistics.scores_dict

    acc_dict = {0: accuracy1, 1: accuracy0, 'overall': accuracy}
    pred_dict = {0: mean_pred0, 1: mean_pred1,'overall': mean_pred}
    results_dict = {'acc_dict': acc_dict, 'pred_dict': pred_dict, 'util': util, 'DP': DP, 'EOP': EOP}

    evaluation_path = "{}/evaluation.json".format(base_save_path)
    save_dictionary(results_dict, evaluation_path )


    decisions_path = "{}/decisions.json".format(base_save_path)
    save_dictionary(scores, decisions_path)

    # ------ compare stats-------

    decisions = statistics.scores_array
    a_test = statistics.a_test.to_frame().to_numpy()
    y_test = statistics.y_test.to_frame().to_numpy()
    updates = len(statistics.acc_list_overall)

    print('---- Floyds stats ----')
    # ------ statistics from Floyd -------
    floyds_stats = Statistics(
        predictions=decisions,
        protected_attributes=a_test,
        ground_truths=y_test,
        additonal_measures={UTILITY: {'measure_function': lambda s, y, decisions: np.mean(decisions * (y - 0.5)),
                                      'detailed': False}})

    save_and_plot_results(
        base_save_path=base_save_path,
        statistics=floyds_stats, update_iterations=updates)

    print('---- My stats ----')
    # my stats
    statistics.save_plot_process_results(results_dict, base_save_path)

    return RegressionPolicy(expgrad_XA), results_dict


