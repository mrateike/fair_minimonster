import os
import sys

root_path = os.path.abspath(os.path.join("."))
if root_path not in sys.path:
    sys.path.append(root_path)

# from contextual_bandit import Runtime
from src.contextual_bandit.Runtime import play
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-T', '--total_data', type=int, nargs='+', required=True,
                        help='list of total data s to be used')
    parser.add_argument('-a', '--alpha', type=float, nargs='+', required=True,
                        help='phase 1 phase 2 data split parameter')
    parser.add_argument('-s', '--seeds', type=int, nargs='+', required=False,
                        help='seeds for phase 1, 2, testing', default=967)

    parser.add_argument('-f', '--fairness_type', type=str, nargs='+', required=True,
                        help="select the type of fairness (DP, EO)")
    parser.add_argument('-bt', '--batch_type', type=str, nargs='+', required=True,
                        help='batches type used (no_batch, exp, lin, warm_start)')
    parser.add_argument('-bs', '--batch_size', type=str, nargs='+', required=False, default=200,
                        help='batches size used for lin (required) otherwise ignored')

    parser.add_argument('-eps', '--eps', type=float, nargs='+', required=True,
                        help="list of statistical unfairness paramenters (beta) to be used")
    parser.add_argument('-nu', '--nu', type=float, nargs='+', required=True,
                        help="list of accuracy parameters of the oracle to be used")
    parser.add_argument('-mu', '--mu', type=float, nargs='+', required=True,
                        help="minimum probability for simulating the bandit")

    # Configuration parameters
    parser.add_argument('-d', '--data', type=str, nargs='+', required=True,
                        help="select the distribution (FICO, Uncalibrated)")

    parser.add_argument('-i', '--iterations', type=str, nargs='+', required=True,
                        help="number of iterations of the bandit coordinate decent algo")
    parser.add_argument('-p', '--path', type=str, required=False, help="save path for the results")


    args = parser.parse_args()



    base_save_path = args.path
    eps = "eps_{}".format((args.eps[0]))
    mu = "mu_{}".format((args.mu[0]))
    nu = "nu_{}".format((args.nu[0]))
    N = "N_{}".format((args.total_data[0]))
    i = "it_{}".format((args.iterations[0]))

    base_save_path = "{}/{}_{}_{}_{}_{}".format(base_save_path, eps, mu, nu, N, i)
    Path(base_save_path).mkdir(parents=True, exist_ok=True)

    T = 5071
    # training data
    TT = 5000
    # testing data
    # TT = N - T
    # phase 1 phase 2 data
    T1 = round(T ** (2 * args.alpha[0]))
    T2 = T - T1


    fairness = args.fairness_type[0]
    batch = args.batch_type[0]
    batchsize = args.batch_size[0]
    eps = args.eps[0]
    nu = args.nu[0]
    dataset = args.data[0]
    seed = args.seeds[0]
    mu = args.mu[0]

    num_iterations = args.iterations[0]
    print('Im running')
    play(T, args.alpha[0], TT, fairness, batch, batchsize, eps, nu, dataset, base_save_path, seed, mu, num_iterations)
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

