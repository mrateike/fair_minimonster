# Copyright (c) 2020 mrateike
import os
import sys

root_path = os.path.abspath(os.path.join(''))
if root_path not in sys.path:
    sys.path.append(root_path)

import statistics
import pandas as pd

_L0 = "l0"
_L1 = "l1"
from src.fairlearn.metrics import accuracy_score_group_summary, \
    demographic_parity_difference, true_positive_rate_difference, utility
from data.util import save_dictionary
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import tikzplotlib as tpl

"""
Evaluation class for computing and saving fairness, accuracy and regret
Includes a plot function
"""



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


        x_test, self.a_test, self.y_test = B.sample_test_dataset(TT,seed)
        self.XA_test = pd.concat([x_test,self.a_test], axis = 1)


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


        if plot_dict['evaluation'] == 'YES':
            if current_column == 0 and current_row == 1:
                axis.set_ylim(-0.5, 0.5)
            else:
                axis.set_ylim(0, 1)


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

