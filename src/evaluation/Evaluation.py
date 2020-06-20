

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
# from data.uncalibrated_score import UncalibratedScore
from data.util import stack, serialize_dictionary, save_dictionary
from src.evaluation.plotting import plot_median, plot_mean
from src.evaluation.training_evaluation import UTILITY
import matplotlib
from data.util import get_list_of_seeds
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from src.evaluation.training_evaluation import Statistics


import tikzplotlib as tpl



class Evaluation(object):
    def __init__(self, TT, seed, path, B):
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
        self.path = path


        x_test, self.a_test, self.y_test = B.sample_test_dataset(TT, seed)
        self.XA_test = pd.concat([x_test,self.a_test], axis = 1)
        test_data = {'A_test': self.a_test.tolist(), 'Y_test': self.y_test.tolist()}
        test_path = "{}test_data.json".format(path)
        save_dictionary(test_data, test_path)

    def evaluate(self, pi, x_axis):

        dec, _ = pi.predict(self.XA_test)
        scores = pd.Series(dec, name="scores_expgrad_XA")

        results_dict = self.get_stats(self.y_test, scores, self.a_test)
        self.save_stats(results_dict, scores, x_axis)

    def evaluate_scores(self, scores, x_axis):
        results_dict = self.get_stats(self.y_test, scores, self.a_test)
        self.save_stats(results_dict, scores, x_axis)


    def get_stats(self, y_test, scores, A_test):

        # -------- definition -----------
        def summary_as_df(name, summary):
            a = pd.Series(summary.by_group)
            a['overall'] = summary.overall
            return pd.DataFrame({name: a})

        # -------- my statistics -----------
        acc = summary_as_df(
            "accuracy_XA",
            accuracy_score_group_summary(y_test, scores, sensitive_features=A_test))


        mean_pred = summary_as_df(
            "acceptance_rate_XA",
            mean_prediction_group_summary(y_test, scores, sensitive_features=A_test))


        util = utility(y_test, scores)

        DP = demographic_parity_difference(y_test, scores, sensitive_features=A_test)
        ratio_DP = demographic_parity_ratio(y_test, scores, sensitive_features=A_test)

        TPR = true_positive_rate_difference(y_test, scores, sensitive_features=A_test)
        ratio_TPR = true_positive_rate_ratio(y_test, scores, sensitive_features=A_test)

        FPR = false_positive_rate_difference(y_test, scores, sensitive_features=A_test)
        # ratio_FPR = false_positive_rate_ratio(y_test, scores, sensitive_features=A_test)

        EO = equalized_odds_difference(y_test, scores, sensitive_features=A_test)
        # ratio_EO = equalized_odds_ratio(y_test, scores, sensitive_features=A_test)

        # # print("--- EVALUATION  ---")
        # classifier_summary = pd.concat([acc, mean_pred], axis=1)
        # display(classifier_summary)
        #
        # print("DP = ", DP)
        # print("DP_ratio = ", ratio_DP)
        # print("EO = ", TPR)
        # print("EO_ratio = ", ratio_TPR)

        results_dict = {'ACC': acc, 'MEAN_PRED': mean_pred, 'DP': DP, 'FPR': FPR, 'TPR':TPR, 'EO': EO, 'UTIL': util}

        return results_dict




    def save_stats(self, results_dict, scores, x_axis):

        acc = results_dict['ACC']
        mean_pred = results_dict['MEAN_PRED']
        parity = results_dict['DP']
        FPR = results_dict['FPR']
        TPR = results_dict['TPR']
        EO = results_dict['EO']
        util = results_dict['UTIL']


        # ------ update lists -----
        self.acc_list_overall.append(float(acc.loc['overall']))
        self.acc_list_0.append(float(acc.loc[0]))
        self.acc_list_1.append(float(acc.loc[1]))
        self.mean_pred_0_list.append(float(mean_pred.loc[0]))
        self.mean_pred_1_list.append(float(mean_pred.loc[1]))
        self.mean_pred_overall_list.append(float(mean_pred.loc['overall']))
        self.DP_list.append(parity)
        self.FPR_list.append(FPR)
        self.TPR_list.append(TPR)
        self.EO_list.append(EO)
        self.util_list.append(util)
        self.scores_dict.update({self.i_scores: scores.tolist()})
        self.i_scores +=1
        if self.scores_array is None:
            self.scores_array = np.array([scores]).T
        else:
            self.scores_array = np.concatenate((self.scores_array, np.array([scores]).T), axis=1)\

        # ------ save lists ----
        acc = self.acc_list_overall
        accuracy0 = self.acc_list_0
        accuracy1 = self.acc_list_1
        mean_pred0 = self.mean_pred_0_list
        mean_pred1 = self.mean_pred_1_list
        mean_pred = self.mean_pred_overall_list
        DP = self.DP_list
        EOP = self.TPR_list
        util = self.util_list

        acc_dict = {0: accuracy1, 1: accuracy0, 'overall': acc}
        pred_dict = {0: mean_pred0, 1: mean_pred1, 'overall': mean_pred}
        results_dict = {'acc_dict': acc_dict, 'pred_dict': pred_dict, 'util': util, 'DP': DP, 'EOP': EOP}

        evaluation_path = "{}measures.json".format(self.path)
        save_dictionary(results_dict, evaluation_path)

        # ---- save decisions
        scores = self.scores_dict
        decisions_path = "{}decisions.json".format(self.path)
        save_dictionary(scores, decisions_path)

        # ---- plot four measures over iterations -----
        plot_dict = {}
        plot_dict['x_axis_test'] = x_axis
        plot_dict['x_label'] = 'individuals'
        plot_dict['y_label0'] = 'util'
        plot_dict['y_label1'] = 'acc'
        plot_dict['y_label2'] = 'DP'
        plot_dict['y_label3'] = 'EOP'
        plot_dict['square'] = 'NO'
        plot_dict['process'] = 'NO'
        plot_dict['evaluation'] = 'YES'
        my_plot(self.path, plot_dict, util, acc, DP, EOP)

        acc_av_dict = save_mean_std_quantiles(acc)
        util_av_dict = save_mean_std_quantiles(util)
        DP_av_dict = save_mean_std_quantiles(DP)
        EOP_av_dict = save_mean_std_quantiles(EOP)

        data_mean = {'UTIL': util_av_dict, 'ACC': acc_av_dict, 'DP': DP_av_dict, 'EOP': EOP_av_dict}

        parameter_save_path = "{}evaluation.json".format(self.path)
        save_dictionary(data_mean, parameter_save_path)

        # # print('---- Floyds stats ----')
        # decisions = self.scores_array
        # a_test = self.a_test.to_frame().to_numpy()
        # y_test = self.y_test.to_frame().to_numpy()
        # updates = len(self.acc_list_overall)

        # # ------ statistics from Floyd -------
        # floyds_stats = Statistics(
        #     predictions=decisions,
        #     protected_attributes=a_test,
        #     ground_truths=y_test,
        #     additonal_measures={UTILITY: {'measure_function': lambda s, y, decisions: np.mean(decisions * (y - 0.5)),
        #                                   'detailed': False}})
        # save_and_plot_results(
        #     base_save_path=self.path,
        #     statistics=floyds_stats, update_iterations=updates)


    # def save_plot_process_results(self, results_dict, path):
    #
    #     acc = results_dict['acc_dict']['overall']
    #     util = results_dict['util']
    #     DP = results_dict['DP']
    #     EOP = results_dict['EOP']
    #
    #     acc_av_dict = save_mean_std_quantiles(acc)
    #     util_av_dict = save_mean_std_quantiles(util)
    #     DP_av_dict = save_mean_std_quantiles(DP)
    #     EOP_av_dict = save_mean_std_quantiles(EOP)
    #
    #     data_mean = {'UTIL': util_av_dict, 'ACC': acc_av_dict , 'DP' : DP_av_dict, 'EOP' :EOP_av_dict}
    #     parameter_save_path = "{}/evaluation_mean.json".format(path)
    #     save_dictionary(data_mean, parameter_save_path)

def save_mean_std_quantiles(measure):

    mean = np.mean(measure)
    FQ = np.percentile(measure, q=25)
    TQ = np.percentile(measure, q=75)
    STD = np.std(measure)
    Q025 = np.quantile(measure, 0.025)
    Q975 = np.quantile(measure, 0.975)

    dict = {}
    dict['mean'] = mean
    dict['FQ'] = FQ
    dict['TQ'] = TQ
    dict['TQ'] = TQ
    dict['STD'] = STD
    dict['Q025'] = Q025
    dict['Q975'] = Q975

    return dict



def get_average_regret(regret_dict):

    regt = regret_dict[list(regret_dict.keys())[0]]
    regt_cum = regret_dict[list(regret_dict.keys())[1]]
    regT = regret_dict[list(regret_dict.keys())[2]]
    regT_cum = regret_dict[list(regret_dict.keys())[3]]

    # Rt = regt_cum[-1]

    RT = regT_cum[-1]
    regt_av_dict = save_mean_std_quantiles(regt)
    regt_cum_av_dict = save_mean_std_quantiles(regt_cum)
    regT_av_dict = save_mean_std_quantiles(regT)
    regT_cum_av_dict = save_mean_std_quantiles(regT_cum)

    av_reg_dict = {'RT': RT, 'regt': regt_av_dict, 'regt_cum': regt_cum_av_dict, 'regT': regT_av_dict, 'regT_cum': regT_cum_av_dict}
    return av_reg_dict



# def my_plot(base_save_path, utility, accuracy, DP, EOP):
#
#     x_label = "time steps"
#     measure_dict = {'utility': utility, 'accuracy':accuracy, 'demographic parity':DP, "true positive rate" : EOP}
#
#     num_columns = 2
#     num_rows = 2
#
#     figure = plt.figure(constrained_layout=True)
#     grid = GridSpec(nrows=num_rows, ncols=num_columns, figure=figure)
#
#
#     current_row = 0
#     current_column = 0
#
#     for key, value in measure_dict.items():
#         axis = figure.add_subplot(grid[current_row, current_column])
#         axis.plot(value)
#         axis.set_xlabel(x_label)
#         axis.title.set_text(key)
#         axis.set_xscale("linear")
#         if current_column ==0 and current_row == 0 :
#             axis.set_ylim(-0.5, 0.5)
#         else:
#             axis.set_ylim(0, 1)
#         #
#         # axis.fill_between(y_FQ,
#         #                   y_TQ,
#         #                   alpha=0.3,
#         #                   edgecolor='#060080',
#         #                   facecolor='#928CFF')
#
#         # c = 0 < 1
#
#         if current_column == 0 and current_row ==0:
#             current_column =1
#         elif current_column == 1 and current_row ==0:
#             current_row =1
#             current_column=0
#         else:
#             current_column = 1
#             current_row = 1
#
#
#     file_path = "{}plot.png".format(base_save_path)
#
#     plt.savefig(file_path)
#     tpl.save(file_path.replace(".png", ".tex"),
#              figure=figure,
#              axis_width='\\figwidth',
#              axis_height='\\figheight',
#              tex_relative_path_to_data='.',
#              extra_groupstyle_parameters={"horizontal sep=1.2cm"},
#              extra_axis_parameters={
#                  "scaled y ticks = false, \n yticklabel style = {/pgf/number format/fixed, /pgf/number format/precision=3}"})
#     plt.close('all')

def my_plot(base_save_path, plot_dict, A1, A2, B1, B2):

    x_label = plot_dict['x_label']

    measure_dict = {plot_dict['y_label0']: A1, plot_dict['y_label1']: A2, plot_dict['y_label2']:B1, plot_dict['y_label3']: B2}

    num_columns = 2
    num_rows = 2

    figure = plt.figure(constrained_layout=True)
    grid = GridSpec(nrows=num_rows, ncols=num_columns, figure=figure)


    current_row = 0
    current_column = 0

    for key, value in measure_dict.items():

        axis = figure.add_subplot(grid[current_row, current_column])

        if plot_dict['process'] == 'YES':
            if current_column == 0:
                x = plot_dict['x_axis_time']
            else:
                x = plot_dict['x_axis_reg']
            axis.plot(x, value)
        else:
            x = plot_dict['x_axis_test']
            axis.plot(x, value)

        axis.set_xlabel(x_label)
        axis.title.set_text(key)
        axis.set_xscale("linear")

        if plot_dict['square'] == 'YES':
            if current_column ==1:
                axis.axis('square')

        if plot_dict['evaluation'] == 'YES':
            if current_column == 0 and current_row == 0:
                axis.set_ylim(-0.5, 0.5)
            else:
                axis.set_ylim(0, 1)

        # axis.fill_between(y_FQ,
        #                   y_TQ,
        #                   alpha=0.3,
        #                   edgecolor='#060080',
        #                   facecolor='#928CFF')

        # c = 0 < 1

        # if plot_dict['evaluation'] == 'YES':
        #     axis.set_ylim([-1,1])

        if current_column == 0 and current_row ==0:
            current_column =1
        elif current_column == 1 and current_row ==0:
            current_row =1
            current_column=0
        else:
            current_column = 1
            current_row = 1

    # Todo: new inserted


    file_path = "{}plot.png".format(base_save_path)
    plt.savefig(file_path)
    tpl.save(file_path.replace(".png", ".tex"),
             figure=figure,
             axis_width='\\figwidth',
             axis_height='\\figheight',
             tex_relative_path_to_data='.',
             extra_groupstyle_parameters={"horizontal sep=1.2cm"})
             # ,extra_axis_parameters={
             #     "scaled y ticks = false, \n yticklabel style = {/pgf/number format/fixed, /pgf/number format/precision=3}"})
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
              file_path="{}/floyds_plot.png".format(base_save_path))

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
