

import os
import sys

root_path = os.path.abspath(os.path.join(''))
if root_path not in sys.path:
    sys.path.append(root_path)

import statistics
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
    def __init__(self, TT, seed, path, B, x_label):
        self.x_label = x_label
        self.DP_list = []
        self.TPR_list = []
        self.ACC_list = []
        self.UTIL_list = []

        self.stats_list = []
        self.scores_dict = {}
        self.scores_array = None
        self.i_scores = 0
        self.path = path

        # randomstate = np.random.RandomState(seed)
        x_test, self.a_test, self.y_test = B.sample_test_dataset(TT,seed)
        # print('TEST DATA', x_test, self.a_test, self.y_test)
        self.XA_test = pd.concat([x_test,self.a_test], axis = 1)
        # test_data = {'A_test': self.a_test.tolist(), 'Y_test': self.y_test.tolist()}
        # test_path = "{}test_data.json".format(path)
        # save_dictionary(test_data, test_path)

    def evaluate(self, pi):

        dec, _ = pi.predict(self.XA_test)
        scores = pd.Series(dec, name="scores_expgrad_XA")

        results_dict = self.get_stats(self.y_test, scores, self.a_test)
        self.save_stats(results_dict)

    def evaluate_scores(self, scores):
        results_dict = self.get_stats(self.y_test, scores, self.a_test)
        self.save_stats(results_dict)


    def get_stats(self, y_test, scores, A_test):

        ACC = accuracy_score_group_summary(y_test, scores, sensitive_features=A_test).overall
        UTIL = utility(y_test, scores)
        DP = demographic_parity_difference(y_test, scores, sensitive_features=A_test)
        TPR = true_positive_rate_difference(y_test, scores, sensitive_features=A_test)
        results_dict = {'ACC': ACC, 'DP': DP, 'TPR': TPR, 'UTIL':UTIL}

        return results_dict


    def save_stats(self, results_dict):

        ACC = results_dict['ACC']
        DP = results_dict['DP']
        TPR = results_dict['TPR']
        UTIL = results_dict['UTIL']

        self.ACC_list.append(ACC)
        self.DP_list.append(DP)
        self.TPR_list.append(TPR)
        self.UTIL_list.append(UTIL)



        # self.scores_dict.update({self.i_scores: scores.tolist()})
        # self.i_scores +=1
        #
        # scores = self.scores_dict
        # decisions_path = "{}decisions.json".format(self.path)
        # save_dictionary(scores, decisions_path)


def get_mean_statistic(stats):
    ACC_mean = statistics.mean(stats.ACC_list)
    DP_mean = statistics.mean(stats.DP_list)
    TPR_mean = statistics.mean(stats.TPR_list)

    if len(stats.ACC_list) == 1:
        ACC_std = 0
        DP_std = 0
        TPR_std = 0
    else:
        ACC_std = statistics.stdev(stats.ACC_list)
        DP_std = statistics.stdev(stats.DP_list)
        TPR_std = statistics.stdev(stats.TPR_list)


    data_mean = {'ACC_mean': ACC_mean, 'ACC_std': ACC_std, \
                 'DP_mean': DP_mean, 'DP_std': DP_std, \
                 'TPR_mean': TPR_mean, 'TPR_std': TPR_std, 'num_policies': len(stats.ACC_list)}

    parameter_save_path = "{}evaluation.json".format(stats.path)
    save_dictionary(data_mean, parameter_save_path)

    # ---- plot four measures over iterations -----

    # x_axis = range(0, len(stats.ERR_list))
    # plot_dict = {}
    # plot_dict['x_axis_test'] = x_axis
    # plot_dict['x_label'] = stats.x_label
    # plot_dict['y_label0'] = 'ERR'
    # plot_dict['y_label1'] = 'DP'
    # plot_dict['y_label2'] = 'UTIL'
    # plot_dict['y_label3'] = 'TPR'
    # plot_dict['process'] = 'NO'
    # plot_dict['evaluation'] = 'YES'
    # my_plot(stats.path, plot_dict, stats.ERR_list, stats.DP_list, stats.UTIL_list, stats.TPR_list)


def get_average_regret(regret_dict):


    regt_cum = regret_dict[list(regret_dict.keys())[1]]
    regT_cum = regret_dict[list(regret_dict.keys())[3]]

    RT = regT_cum[-1]


    av_reg_dict = {'RT': RT, 'regt': regt_av_dict, 'regt_cum': regt_cum_av_dict, 'regT': regT_av_dict, 'regT_cum': regT_cum_av_dict}
    return av_reg_dict


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

        # if plot_dict['square'] == 'YES':
        #     if current_column ==1:
        #         axis.axis('square')

        if plot_dict['evaluation'] == 'YES':
            if current_column == 0 and current_row == 1:
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


    file_path = "{}plot.png".format(base_save_path)
    plt.savefig(file_path)
    tpl.save(file_path.replace(".png", ".tex"),
             figure=figure,
             axis_width='\\figwidth',
             axis_height='\\figheight',
             tex_relative_path_to_data='.',
             extra_groupstyle_parameters={"horizontal sep=1.2cm"})
    plt.close('all')

