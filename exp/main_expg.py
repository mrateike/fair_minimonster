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
from src.evaluation.Evaluation import Evaluation, my_plot, get_mean_statistic
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

    parser.add_argument('-T', '--total_data', type=int, nargs='+', required=True,
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

    T = args.total_data[0]
    TT = args.total_data[0]
    T1 = round(T ** (2 * args.alpha[0]))
    T2 = T - T1



    base_save_path = args.path
    eps = "eps_{}".format((args.eps[0]))
    mu = "mu_{}".format((args.mu[0]))
    nu = "nu_{}".format((args.nu[0]))
    T2 = "T2_{}".format(T2)

    base_save_path = "{}/{}-{}-{}-{}".format(base_save_path, eps, mu, nu, T2)
    Path(base_save_path).mkdir(parents=True, exist_ok=True)


    seed_test = args.seeds[0]*45
    seed_training = 17*args.seeds[0]

    B = Simulators.DatasetBandit(args.data[0])
    dataset = B.sample_dataset(T, seed_training)
    dataset1 = dataset.iloc[:T1]
    dataset2 = dataset.iloc[T1:T]

    x_label = 'policies'
    base_save_path = "{}/oracle_".format(base_save_path)
    statistics = Evaluation(TT, seed_test, base_save_path, B, x_label)

    if args.fairness_type[0] == 'DP':
        fairness = DemographicParity()
    elif args.fairness_type[0] == 'EO':
        fairness = TruePositiveRateDifference()

    i = 0

    while i < args.iterations:
        # print('I am computing policy ', i)
        play(dataset1, dataset2, fairness, args.eps[0], args.nu[0], statistics, args.mu[0])
        i += 1

    get_mean_statistic(statistics)