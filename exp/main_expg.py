import os
import sys

root_path = os.path.abspath(os.path.join("."))
if root_path not in sys.path:
    sys.path.append(root_path)

from src.fairlearn.reductions._moments.conditional_selection_rate import DemographicParity
from src.fairlearn.reductions._exponentiated_gradient.run_expg import play
from src.evaluation.Evaluation import Evaluation

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Policy training parameters
    parser.add_argument('-T1', '--time_steps_1', type=int, required=True,
                        help='list of phase 1 time steps to be used')
    parser.add_argument('-T2', '--time_steps_2', type=int, required=True,
                        help='list of phase 2 time steps to be used')
    parser.add_argument('-TT', '--time_steps_testing', type=int, required=False,
                        help='testing time steps to be used', default=1000)
    # parser.add_argument('-bs', '--batch_sizes', type=int, nargs='+', required=True,
    #                     help='list of batch sizes to be used')

    # Fairness parameters
    parser.add_argument('-f', '--fairness_type', type=str, required=True,
                        help="select the type of fairness (DP, FPR)"
                             "if none is selected no fairness criterion is applied")
    parser.add_argument('-eps', '--eps',
                        type=float, nargs='+', required=True,
                        help="list of statistical unfairness paramenters to be used")
    parser.add_argument('-nu', '--nu', type=float, nargs='+', required=True,
                        help="list of accuracy parameters of the oracle to be used")

    # Configuration parameters

    parser.add_argument('--plot', required=False, action='store_true')

    args = parser.parse_args()

    # parser = argparse.ArgumentParser(description='Bechavods Fair Minimonster')
    # parser.add_argument('T1', type=int, help='phase 1 time steps')
    # parser.add_argument('T2', type=int, help='phase 2 time steps')
    # parser.add_argument('TT', type=int, help='test set size')
    # parser.add_argument('fairness', type=str, help='fairness: DP')
    # parser.add_argument('batch', type=str, help='batch: exp, lin batchsize, none')
    # args = parser.parse_args()

    acc = []
    DP = []
    TPR = []
    FPR = []
    EO = []

    T = 50
    i = 0
    statistics = Evaluation()

    if args.fairness_type == 'DP':
        fairness = DemographicParity()

    while i < T:
        print('I am computing policy ', i)
        accuracy, mean_pred, util, DP, TPR, EO =  play(args.time_steps_1, args.time_steps_2, args.time_steps_testing, fairness, args.eps, args.nu, statistics)
        i += 1


    fig_train = plt.figure()
    # It's the arrangement of subgraphs within this graph. The first number is how many rows of subplots; the second number is how many columns of subplots; the third number is the subgraph you're talking about now. In this case, there's one row and one column of subgraphs (i.e. one subgraph) and the axes are talking about the first of them. Something like fig.add_subplot(3,2,5) would be the lower-left subplot in a grid of three rows and two columns
    ax1_train = fig_train.add_subplot(111)
    ax1_train.scatter(range(0, len(accuracy)), accuracy, label='accuracy')
    ax1_train.scatter(range(0, len(mean_pred)), mean_pred, label='mean_pred')
    ax1_train.scatter(range(0, len(DP)), DP, label='DP')
    ax1_train.scatter(range(0, len(TPR)), TPR, label='TPR')
    plt.xlabel("policies")
    plt.ylabel("metrics")
    plt.title('Oracle')
    plt.legend()
    plt.show()

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