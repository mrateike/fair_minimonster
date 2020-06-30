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
from data.util import get_list_of_seeds

"""
This is the main function for the fair minimonster algorithm
"""

def play(T, alpha, TT, fairness, batch, batchsize, eps, nu, dataset, path, seed, mu, num_iterations):

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

