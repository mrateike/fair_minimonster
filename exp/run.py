import os
import sys

root_path = os.path.abspath(os.path.join("."))
if root_path not in sys.path:
    sys.path.append(root_path)

# from contextual_bandit import Runtime
from src.contextual_bandit.Runtime import play
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Policy training parameters
    parser.add_argument('-N', '--data_set', type=int, required=True,
                        help='total data (time stepts) used')
    parser.add_argument('-a', '--alpha', type=int, required=True,
                        help='phase1 phase2 splitting parameter')
    # parser.add_argument('-TT', '--time_steps_testing', type=int, required=False,
    #                     help='testing time steps to be used', default=10000)
    # parser.add_argument('-bs', '--batch_sizes', type=int, nargs='+', required=True,
    #                     help='list of batch sizes to be used')

    # Fairness parameters
    parser.add_argument('-f', '--fairness_type', type=str, required=False,
                        help="select the type of fairness (DP, FPR)"
                             "if none is selected no fairness criterion is applied")
    parser.add_argument('-bt', '--batch_type', type=str, required=True,
                        help='batches type used (exp, lin, none)')
    parser.add_argument('-bs', '--batch_size', type=str, required=True,
                        help='batches size used for lin (required)')
    parser.add_argument('-eps', '--eps',
                        type=float, nargs='+', required=True,
                        help="list of statistical unfairness paramenters to be used")
    parser.add_argument('-nu', '--nu', type=float, nargs='+', required=True,
                        help="list of accuracy parameters of the oracle to be used")

    # Configuration parameters
    parser.add_argument('-d', '--data', type=str, required=True,
                        help="select the distribution (FICO, COMPAS, ADULT, GERMAN, Uncalibrated)")
    parser.add_argument('-p', '--path', type=str, required=True, help="save path for the results")

    parser.add_argument('--plot', required=False, action='store_true')

    args = parser.parse_args()



    # parser = argparse.ArgumentParser(description='Bechavods Fair Minimonster')
    # parser.add_argument('T1', type=int, help='phase 1 time steps')
    # parser.add_argument('T2', type=int, help='phase 2 time steps')
    # parser.add_argument('TT', type=int, help='test set size')
    # parser.add_argument('fairness', type=str, help='fairness: DP')
    # parser.add_argument('batch', type=str, help='batch: exp, lin batchsize, none')
    # args = parser.parse_args()


    N = args.data_set
    # training data
    T = round(0.75 * T)
    # testing data
    TT = N-T

    T1 = round(T**(args.alpha))
    T2 = T - T1

    print('N', N)
    print('T1', T2)
    print('T2', T2)
    print('TT', TT)

    # T1 = args.time_steps_1
    # T2 = args.time_steps_2
    # TT = args.time_steps_testing
    fairness = args.fairness_type
    batch = args.batch_type
    batchsize = args.batch_size
    eps = args.eps
    nu = args.nu
    dataset = args.data

    print('Im running')
    play(T1, T2, TT, fairness, batch, batchsize, eps, nu, dataset, args.path)
    print('/////// FINISHED ///////////')


# without args
# T1 = 100
# T2 = 1000
# TT = 1000
# fairness = 'DP'
# batch = 'lin 500'
# # (FICO, COMPAS, ADULT, GERMAN, TOY), here: Toy is Uncalibrated, to do: change
# dataset = 'Uncalibrated'

# batch can take values 'exp', 'lin batchsize', 'none'

# eps = 0.01
# nu = 1e-6

