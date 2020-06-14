

import os
import sys

root_path = os.path.abspath(os.path.join(''))
if root_path not in sys.path:
    sys.path.append(root_path)


import pandas as pd
import numpy as np
from IPython.display import display
import json
import inspect
import numbers
from src.evaluation.training_evaluation import Statistics
_L0 = "l0"
_L1 = "l1"
from src.fairlearn.metrics import mean_prediction_group_summary, accuracy_score_group_summary, \
    equalized_odds_difference, demographic_parity_difference, demographic_parity_ratio, \
    equalized_odds_ratio, true_positive_rate_difference, true_positive_rate_ratio, \
    false_positive_rate_difference, false_positive_rate_ratio, utility
# import matplotlib.pyplot as plt
from data.uncalibrated_score import UncalibratedScore
from data.util import stack, serialize_dictionary, save_dictionary
from src.evaluation.plotting import plot_median, plot_mean
from src.evaluation.training_evaluation import UTILITY
import matplotlib
from data.util import get_list_of_seeds
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import tikzplotlib as tpl



class Evaluation(object):
    def __init__(self, TT, seed):
        # self.acc_dict ={'overall':{}, 0:{}, 1:{}}
        # self.mean_pred_dict = {'overall':{}, 0:{}, 1:{}}
        # set seed

        self.DP_list = []
        self.TPR_list = []
        self.FPR_list = []
        self.EOP_list = []
        self.EO_list = []
        self.acc_list_overall = []
        self.acc_list_0 = []
        self.acc_list_1 = []
        self.mean_pred_overall_list = []
        self.mean_pred_0_list = []
        self.mean_pred_1_list = []
        self.util_list = []
        self.stats_list = []
        self.scores_dict = {}
        self.scores_array = None
        self.i_scores = 0

        fraction_protected = 0.5
        distribution = UncalibratedScore(fraction_protected)
        x_test, a_test, y_test = distribution.sample_test_dataset(TT, seed)
        x_test = pd.DataFrame(x_test.squeeze())
        self.y_test = pd.Series(y_test.squeeze(), name='label')
        a_test = pd.Series(a_test.squeeze(), name='sensitive_features_X')
        self.XA_test = pd.concat([x_test, a_test == 1], axis=1).astype(float)
        self.a_test = a_test.rename('sensitive_features')



    def evaluate(self, pi):
        print('--- validation oracle policy returned --- ')

        # get prediction
        dec_prob = pi.predict(self.XA_test)
        scores = pd.Series(dec_prob[:,0], name="scores_expgrad_XA").astype(int)

        # cannot merge these two
        # Todo: decisions

        results_dict = self.get_stats(self.y_test, scores, self.a_test)
        self.save_stats(results_dict, scores)


    def get_stats(self, y_test, scores, A_test):

        # -------- my statistics -----------
        def summary_as_df(name, summary):
            a = pd.Series(summary.by_group)
            a['overall'] = summary.overall
            return pd.DataFrame({name: a})

        acc = summary_as_df(
            "accuracy_XA",
            accuracy_score_group_summary(y_test, scores, sensitive_features=A_test))

        mean_pred = summary_as_df(
            "acceptance_rate_XA",
            mean_prediction_group_summary(y_test, scores, sensitive_features=A_test))


        util = utility(y_test, scores)

        DP = demographic_parity_difference(y_test, scores, sensitive_features=A_test)
        ratio_DP = demographic_parity_ratio(y_test, scores, sensitive_features=A_test)

        FPR = false_positive_rate_difference(y_test, scores, sensitive_features=A_test)
        ratio_FPR = false_positive_rate_ratio(y_test, scores, sensitive_features=A_test)

        TPR = true_positive_rate_difference(y_test, scores, sensitive_features=A_test)
        ratio_TPR = true_positive_rate_ratio(y_test, scores, sensitive_features=A_test)

        EO = equalized_odds_difference(y_test, scores, sensitive_features=A_test)
        ratio_EO = equalized_odds_ratio(y_test, scores, sensitive_features=A_test)

        EOP = FPR + TPR


        print("--- EVALUATION  ---")
        classifier_summary = pd.concat([acc, mean_pred], axis=1)
        display(classifier_summary)

        print("DP = ", DP)
        print("DP_ratio = ", ratio_DP)
        print("FPR = ", FPR)
        print("FPR_ratio = ", ratio_FPR)
        print("TPR = ", TPR)
        print("TPR_ratio = ", ratio_TPR)
        print("EO = ", EO)
        print("EO_ratio = ", ratio_EO)

        results_dict= {'ACC': acc, 'MEAN_PRED':mean_pred, 'DP': DP, 'FPR': FPR, 'TPR':TPR, 'EO':EO, 'UTIL':util, 'EOP':EOP}
        return results_dict

    def save_stats(self, results_dict, scores):
        acc = results_dict['ACC']
        mean_pred = results_dict['MEAN_PRED']
        parity = results_dict['DP']
        FPR = results_dict['FPR']
        TPR = results_dict['TPR']
        EO = results_dict['EO']
        util = results_dict['UTIL']
        EOP = results_dict['EOP']

        self.acc_list_overall.append(float(acc.loc['overall']))
        self.acc_list_0.append(float(acc.loc[0]))
        self.acc_list_1.append(float(acc.loc[1]))
        self.mean_pred_overall_list.append(float(mean_pred.loc['overall']))
        self.mean_pred_0_list.append(float(mean_pred.loc[0]))
        self.mean_pred_1_list.append(float(mean_pred.loc[1]))
        self.DP_list.append(parity)
        self.FPR_list.append(FPR)
        self.TPR_list.append(TPR)
        self.EOP_list.append(EOP)
        self.EO_list.append(EO)
        self.util_list.append(util)
        self.scores_dict.update({self.i_scores: scores.tolist()})
        self.i_scores +=1

        if self.scores_array is None:
            self.scores_array = np.array([scores]).T
        else:
            self.scores_array = np.concatenate((self.scores_array, np.array([scores]).T), axis=1)\

    def save_plot_process_results(self, results_dict, path):

        acc = results_dict['acc_dict']['overall']
        util = results_dict['util']
        DP = results_dict['DP']
        # Todo: change to EOP, if works
        EOP = results_dict['EOP']


        my_plot(path, util, acc, DP, EOP)

        acc_mean = np.mean(acc)
        util_mean = np.mean(util)
        DP_mean = np.mean(DP)
        EOP_mean = np.mean(EOP)



        # Computations over T policy results


        acc_FQ = np.percentile(acc, q=25)
        acc_TQ = np.percentile(acc, q=75)
        util_FQ = np.percentile(util, q=25)
        util_TQ = np.percentile(util, q=75)
        DP_FQ = np.percentile(DP, q=25)
        DP_TQ = np.percentile(DP, q=75)
        EOP_FQ = np.percentile(EOP, q=25)
        EOP_TQ = np.percentile(EOP, q=75)

        acc_STD = np.std(acc)
        util_STD = np.std(util)
        DP_STD = np.std(DP)
        EOP_STD = np.std(EOP)

        acc_Q025 = np.quantile(acc, 0.025)
        acc_Q975 = np.quantile(acc, 0.975)
        util_Q025 = np.quantile(util, 0.025)
        util_Q975 = np.quantile(util, 0.975)
        DP_Q025 = np.quantile(DP, 0.025)
        DP_Q975 = np.quantile(DP, 0.975)
        EOP_Q025 = np.quantile(EOP, 0.025)
        EOP_Q975 = np.quantile(EOP, 0.975)

        data_mean = {'UTIL_mean': util_mean, 'UTIL_FQ': util_FQ, 'UTIL_TQ': util_TQ, \
                     'ACC_mean': acc_mean, 'ACC_FQ': acc_FQ, 'ACC_TQ': acc_TQ, \
                     'DP_mean': DP_mean, 'DP_FQ': DP_FQ, 'DP_TQ': DP_TQ, \
                     'EOP_mean': EOP_mean, 'EOP_FQ': EOP_FQ, 'EOP_TQ': EOP_TQ, \
                     'UTIL_STD': util_STD, 'ACC_STD': acc_STD, 'DP_STD': DP_STD, 'EOP_STD': EOP_STD, \
                     'UTIL_Q025': util_Q025, 'UTIL_Q975': util_Q975, \
                     'ACC_Q025': acc_Q025, 'ACC_Q975': acc_Q975, \
                     'DP_Q025': DP_Q025, 'DP_Q975': DP_Q975, \
                     'EOP_Q025': EOP_Q025, 'EOP_Q975': EOP_Q975}

        parameter_save_path = "{}/evaluation_mean.json".format(path)
        save_dictionary(data_mean, parameter_save_path)


def my_plot(base_save_path, utility, accuracy, DP, EOP):

    #x_scale = plotting_dictionary["plot_info"]["x_scale"]
    x_label = "time steps"
    measure_dict = {'utility': utility, 'accuracy':accuracy, 'demographic parity':DP, "false positive rate" : EOP}

    num_columns = 2
    num_rows = 2

    figure = plt.figure(constrained_layout=True)
    grid = GridSpec(nrows=num_rows, ncols=num_columns, figure=figure)


    current_row = 0
    current_column = 0

    for key, value in measure_dict.items():
        # y = y = y_dict["mean"]
        # y_FQ = y_dict["FQ"]
        # y_TQ = y_dict["TQ"]

        axis = figure.add_subplot(grid[current_row, current_column])
        axis.plot(value)
        axis.set_xlabel(x_label)
        axis.title.set_text(key)
        axis.set_xscale("linear")
        #
        # axis.fill_between(y_FQ,
        #                   y_TQ,
        #                   alpha=0.3,
        #                   edgecolor='#060080',
        #                   facecolor='#928CFF')

        # c = 0 < 1
        if current_column == 0 and current_row ==0:
            current_column =1
        elif current_column == 1 and current_row ==0:
            current_row =1
            current_column=0
        else:
            current_column = 1
            current_row = 1

        # if current_column > 0:
        #     current_row += 1
        #     current_column = 0

    file_path = "{}/my_results.png".format(base_save_path)
    plt.savefig(file_path)
    tpl.save(file_path.replace(".png", ".tex"),
             figure=figure,
             axis_width='\\figwidth',
             axis_height='\\figheight',
             tex_relative_path_to_data='.',
             extra_groupstyle_parameters={"horizontal sep=1.2cm"},
             extra_axis_parameters={
                 "scaled y ticks = false, \n yticklabel style = {/pgf/number format/fixed, /pgf/number format/precision=3}"})
    plt.close('all')


def save_and_plot_results(base_save_path, statistics, update_iterations):
    """ Stores the training results (statistics and/or model parameters) in the specified path.
    from Floyds snippets

    Args:
        base_save_path: The base path specified by the user under which the results will be stored.
        statistics: The statistics of either a specific training run for a specified lambda, or the overall statistics over all lambdas.
        model_parameters: The parameters of a model for a specific lambda.
        sub_directory: The subdirectory under which the results will be stored.
    """
    # save the model parameters to be able to restore the model

    save_path = "{}/".format(base_save_path)

    # save the results for each lambda
    statistics_save_path = "{}statistics.json".format(save_path)
    serialized_statistics = statistics.to_dict()
    save_dictionary(serialized_statistics, statistics_save_path)

    plot_mean(x_values=range(update_iterations),
              x_label="Time steps",
              x_scale="linear",
              performance_measures=[statistics.get_additonal_measure(UTILITY, "Utility"),
              # performance_measures=[
                                    statistics.accuracy()],
              fairness_measures=[statistics.demographic_parity(),
                                 statistics.equality_of_opportunity()],
              file_path="{}/results_mean_time.png".format(base_save_path))

#     plot_median(x_values=range(update_iterations),
#                 x_label="Time steps",
#                 x_scale="linear",
#                 performance_measures=[statistics.get_additonal_measure(UTILITY, "Utility"),
#                 # performance_measures=[
#                                       statistics.accuracy()],
#                 fairness_measures=[statistics.demographic_parity(),
#                                    statistics.equality_of_opportunity()],
#                 file_path="{}/results_median_time.png".format(base_save_path))
# #
