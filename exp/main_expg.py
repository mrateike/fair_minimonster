import os
import sys
import time
import numpy as np
from pathlib import Path
root_path = os.path.abspath(os.path.join("."))
if root_path not in sys.path:
    sys.path.append(root_path)

from src.contextual_bandit import Simulators

from src.fairlearn.reductions._moments.conditional_selection_rate import DemographicParity, TruePositiveRateDifference
from src.fairlearn.reductions._exponentiated_gradient.run_expg import play
from src.evaluation.Evaluation import Evaluation, my_plot
from data.util import save_dictionary
from statistics import mean
from data.distribution import UncalibratedScore, FICODistribution
from data.util import get_list_of_seeds


import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-N', '--total_data', type=int, nargs='+', required=True,
                        help='list of toatl data s to be used')
    parser.add_argument('-a', '--alpha', type=float, nargs='+', required=True,
                        help='phase 1 phase 2 data split parameter')
    parser.add_argument('-s', '--seeds', type=int, nargs='+', required=False,
                        help='seeds for phase 1, 2, testing', default=967)
    parser.add_argument('-i', '--iterations', type=int, required=False,
                        help='how many policies should be computed per setting', default=100)
    parser.add_argument('-f', '--fairness_type', type=str, nargs='+', required=True,
                        help="select the type of fairness (DP, EO)")
    parser.add_argument('-eps', '--eps', type=float, nargs='+', required=True,
                        help="list of statistical unfairness paramenters (beta) to be used")
    parser.add_argument('-nu', '--nu', type=float, nargs='+', required=True,
                        help="list of accuracy parameters of the oracle to be used")
    parser.add_argument('-mu', '--mu', type=float, nargs='+', required=True,
                        help="minimum probability for simulating the bandit")
    parser.add_argument('-d', '--data', type=str, nargs='+', required=True,
                        help="select the distribution (FICO, Uncalibrated)")
    parser.add_argument('-p', '--path', type=str, required=True, help="save path for the results")

    args = parser.parse_args()

    base_save_path = args.path
    eps = "eps_{}".format((args.eps[0]))
    mu = "mu_{}".format((args.mu[0]))
    nu = "nu_{}".format((args.nu[0]))
    N = "N_{}".format((args.total_data[0]))

    base_save_path = "{}/{}-{}-{}-{}".format(base_save_path, eps, mu, nu, N)
    Path(base_save_path).mkdir(parents=True, exist_ok=True)

    N = args.total_data[0]
    # training data
    T = round(0.8 * N)
    # testing data
    TT = N - T
    #phase 1 phase 2 data
    T1 = round(T ** (2 * args.alpha[0]))
    T2 = T-T1

    seed_test = args.seeds[0]*45
    seed_training = 17*args.seeds[0]


    B = Simulators.DatasetBandit(args.data[0])
    dataset = B.sample_dataset((T1 + T2), seed_training)
    dataset1 = dataset.iloc[:T1]
    dataset2 = dataset.iloc[T1:(T1 + T2)]

    base_save_path = "{}/oracle_".format(base_save_path)
    statistics = Evaluation(TT, seed_test, base_save_path, B)

    if args.fairness_type[0] == 'DP':
        fairness = DemographicParity()
    elif args.fairness_type[0] == 'EO':
        fairness = TruePositiveRateDifference()

    i = 0

    while i < args.iterations:
        print('I am computing policy ', i)
        # results_dict, decisions =  play(dataset1, dataset2, fairness, args.eps[0], args.nu[0], statistics, args.mu[0])
        play(dataset1, dataset2, fairness, args.eps[0], args.nu[0], statistics, args.mu[0])
        # parameter_save_path = "{}/evaluation.json".format(base_save_path)
        # save_dictionary(results_dict, parameter_save_path)
        # my_plot(base_save_path, results_dict['util'], results_dict['acc_dict']['overall'], results_dict['DP'], results_dict['FPR'])
        #
        # parameter_save_path = "{}/decisions.json".format(base_save_path)
        # save_dictionary(decisions, parameter_save_path)
        i += 1



    # acc = results_dict['acc_dict']['overall']
    # util = results_dict['util']
    # DP = results_dict['DP']
    # FPR = results_dict['FPR']
    #
    # # Computations over T policy results
    # acc_mean = mean(acc)
    # util_mean = mean(util)
    # DP_mean = mean(DP)
    # FPR_mean = mean(FPR)
    #
    # acc_FQ = np.percentile(acc, q=25)
    # acc_TQ = np.percentile(acc, q=75)
    # util_FQ = np.percentile(util, q=25)
    # util_TQ = np.percentile(util, q=75)
    # DP_FQ = np.percentile(DP, q=25)
    # DP_TQ = np.percentile(DP, q=75)
    # FPR_FQ = np.percentile(FPR, q=25)
    # FPR_TQ = np.percentile(FPR, q=75)
    #
    # acc_STD = np.std(acc)
    # util_STD = np.std(util)
    # DP_STD = np.std(DP)
    # FPR_STD = np.std(FPR)
    #
    # acc_Q025 = np.quantile(acc, 0.025)
    # acc_Q975 = np.quantile(acc, 0.975)
    # util_Q025 = np.quantile(util, 0.025)
    # util_Q975 = np.quantile(util, 0.975)
    # DP_Q025 = np.quantile(DP, 0.025)
    # DP_Q975 = np.quantile(DP, 0.975)
    # FPR_Q025 = np.quantile(FPR, 0.025)
    # FPR_Q975 = np.quantile(FPR, 0.975)
    #
    #
    # data_mean = {'UTIL_mean': util_mean, 'UTIL_FQ': util_FQ, 'UTIL_TQ': util_TQ, \
    #                           'ACC_mean': acc_mean, 'ACC_FQ' : acc_FQ , 'ACC_TQ':acc_TQ, \
    #                           'DP_mean':DP_mean, 'DP_FQ':DP_FQ  , 'DP_TQ':DP_TQ , \
    #                           'FPR_mean' : FPR_mean, 'FPR_FQ': FPR_FQ, 'FPR_TQ': FPR_TQ, \
    #                           'UTIL_STD': util_STD, 'ACC_STD': acc_STD, 'DP_STD': DP_STD, 'FPR_STD': FPR_STD, \
    #                           'UTIL_Q025' : util_Q025, 'UTIL_Q975' :util_Q975, \
    #                           'ACC_Q025': acc_Q025, 'ACC_Q975' : acc_Q975, \
    #                           'DP_Q025' : DP_Q025, 'DP_Q975' : DP_Q975, \
    #                            'FPR_Q025' : FPR_Q025, 'FPR_Q975' : FPR_Q975}
    #
    # parameter_save_path = "{}/evaluation_mean.json".format(base_save_path)
    # save_dictionary(data_mean, parameter_save_path)









    # fig_train = plt.figure()
    # # It's the arrangement of subgraphs within this graph. The first number is how many rows of subplots; the second number is how many columns of subplots; the third number is the subgraph you're talking about now. In this case, there's one row and one column of subgraphs (i.e. one subgraph) and the axes are talking about the first of them. Something like fig.add_subplot(3,2,5) would be the lower-left subplot in a grid of three rows and two columns
    # ax1_train = fig_train.add_subplot(111)
    # ax1_train.scatter(range(0, len(accuracy)), accuracy, label='accuracy')
    # ax1_train.scatter(range(0, len(mean_pred)), mean_pred, label='mean_pred')
    # ax1_train.scatter(range(0, len(DP)), DP, label='DP')
    # ax1_train.scatter(range(0, len(TPR)), TPR, label='TPR')
    # plt.xlabel("policies")
    # plt.ylabel("metrics")
    # plt.title('Oracle')
    # plt.legend()
    # plt.show()

    # improved_acc, improved_DP, improved_TPR, improved_FPR, improved_equalOdds \
    #     = play(args.time_steps_1, args.time_steps_2, args.time_steps_testing, fairness, args.eps, args.nu, statistics)
        # play(args.time_steps_1, args.time_steps_2, args.time_steps_testing, fairness, args.eps, args.nu, statistics)
        # acc.append(improved_acc)
        # DP.append(improved_DP)
        # TPR.append(improved_TPR)
        # FPR.append(improved_FPR)
        # EO.append(improved_equalOdds)

        #
        # print(' --- Overall Results ----')
        # print('ACC mean', np.mean(acc))
        # print("DP mean", np.mean(DP))
        #
        # print('ACC var', np.var(acc))
        # print("DP var", np.var(DP))