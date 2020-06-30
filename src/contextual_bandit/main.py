import os
import sys

root_path = os.path.abspath(os.path.join("."))
if root_path not in sys.path:
    sys.path.append(root_path)

from src.fairlearn.reductions._exponentiated_gradient.exponentiated_gradient import ExponentiatedGradient
from src.fairlearn.reductions._moments.conditional_selection_rate import DemographicParity, TruePositiveRateDifference
from matplotlib import pyplot as plt
from src.contextual_bandit import Simulators
from src.contextual_bandit.Minimonster import FairMiniMonster
import pandas as pd
from data.util import save_dictionary

_L0 = "l0"
_L1 = "l1"
import time
from sklearn.linear_model import LogisticRegression
from src.evaluation.Evaluation import Evaluation
import numpy as np
import argparse
from pathlib import Path

"""
This is the main function for the fair minimonster algorithm from the paper 
Bechavod, Y., Ligett, K., Roth, A., Waggoner, B., & Wu, S. Z. (2019). 
Equal opportunity in online classification with partial feedback.
https://arxiv.org/pdf/1902.02242.pdf


--- parameters ----
    T: int  
        total number of rounds
    alpha: float 
        splitting parameter for rounds, i.e., T1 = T^{2alpha}
        min vaue: 0.25, max value: 0.5
    TT: int
        total number of test data to be generated
    fairness: str
        fairness type ('DP' for demograhpic parity, 'EO' for equal opportunity)
    batch: str
        batch type ('none': no batch, 'lin': linear , 'exp': exponential)
    bt : int
        batch size for linear batch otherwise 1
    eps: float
        fairness relaxation parameter, value > 0
    nu: float
        accuracy parameter, value > 0
    dataset : int
        dataset type ('Uncalibrated': synthetic dataset, 'FICO': FICO datset)
    path: str
        path directory to save results
    mu : float
        minimum probability for smoothed distribution
    i: int
        maximum number of iterations of coordinate descent loop
    seed: int
        random seed for fixing training and test data
   
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-T', '--total_data', nargs='+', type=int, required=True,
                        help='list of total data s to be used')
    parser.add_argument('-a', '--alpha', type=float, nargs='+', required=True,
                        help='phase 1 phase 2 data split parameter')
    parser.add_argument('-s', '--seeds', type=int, nargs='+', required=False,
                        help='seeds for phase 1, 2, testing', default=967)
    parser.add_argument('-f', '--fairness_type', type=str, nargs='+', required=True,
                        help="select the type of fairness (DP, EO)")
    parser.add_argument('-bt', '--batch_type', type=str, nargs='+', required=True,
                        help='batches type used (no_batch, exp, lin, warm_start)')
    parser.add_argument('-bs', '--batch_size', type=str, nargs='+', required=False, default=1,
                        help='batches size used for lin (required) otherwise 1')
    parser.add_argument('-eps', '--eps', type=float, nargs='+', required=True,
                        help="list of statistical unfairness paramenters (beta) to be used")
    parser.add_argument('-nu', '--nu', type=float, nargs='+', required=True,
                        help="list of accuracy parameters of the oracle to be used")
    parser.add_argument('-mu', '--mu', type=float, nargs='+', required=True,
                        help="minimum probability for simulating the bandit")
    parser.add_argument('-d', '--data', type=str, nargs='+', required=True,
                        help="select the distribution (FICO, Uncalibrated)")
    parser.add_argument('-i', '--iterations', type=str, nargs='+', required=True,
                        help="number of iterations of the bandit coordinate decent algo")
    parser.add_argument('-p', '--path', type=str, required=False, help="save path for the results")

    args = parser.parse_args()

    base_save_path = args.path
    path = "{}/results".format(base_save_path)
    Path(path).mkdir(parents=True, exist_ok=True)

    T = args.total_data[0]
    TT = T

    fairness = args.fairness_type[0]
    batch = args.batch_type[0]
    batchsize = args.batch_size[0]
    eps = args.eps[0]
    nu = args.nu[0]
    dataset = args.data[0]
    seed = args.seeds[0]
    mu = args.mu[0]
    alpha = args.alpha[0]
    num_iterations = args.iterations[0]


    if fairness == "EO":
        fairness = TruePositiveRateDifference()
    elif fairness == 'DP':
        fairness = DemographicParity()

    B = Simulators.DatasetBandit(dataset)

    dataset = B.sample_dataset(T, seed)

    M = FairMiniMonster(B, fairness, eps, nu, TT, seed, path, mu, num_iterations)

    print("------------- start fit ---------------")
    start = time.time()

    M.fit(dataset, alpha, batch, batchsize)

    stop = time.time()
    training_time = np.array([stop - start])

    print('------------- END of ALGORITHM  ----- time', training_time)

