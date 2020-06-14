from src.fairlearn.reductions._exponentiated_gradient.exponentiated_gradient import ExponentiatedGradient
from src.fairlearn.reductions._moments.conditional_selection_rate import DemographicParity, TruePositiveRateDifference
from matplotlib import pyplot as plt
from src.contextual_bandit import Simulators
from src.contextual_bandit.Minimonster import MiniMonster
import pandas as pd
from data.util import save_dictionary

_L0 = "l0"
_L1 = "l1"
import time
from sklearn.linear_model import LogisticRegression
from src.evaluation.Evaluation import Evaluation
import numpy as np
from data.util import get_list_of_seeds

# class Runtime(object):

def play(T1, T2, TT, fairness, batch, batchsize, eps, nu, dataset, path, seed, mu):


    seed_test = 45*seed
    seed_phase1 = 37*seed
    seed_phase2 = 17*seed


    if fairness == "EO":
        fairness = TruePositiveRateDifference()
    elif fairness == 'DP':
        fairness = DemographicParity()

    B = Simulators.DatasetBandit(dataset)


    dataset1 = B.sample_dataset(T1, seed_phase1)
    dataset2 = B.sample_dataset(T2, seed_phase2)
    dataset2 = dataset2.set_axis(range(T1, T1+T2), axis=0)
    print('dataset1', dataset1)
    print('dataset2', dataset2)

    M = MiniMonster(B, fairness, dataset1, eps, nu, TT, seed_test, dataset2, path, mu)

    print("------------- start fit ---------------")
    start = time.time()

    # input fairness
    reg1, reg2 = M.fit(T2, T1, batch, batchsize)

    stop = time.time()
    training_time = np.array([stop - start])

    plt.plot(reg1)
    fig_path = "{}/regret1_nonzero.png".format(path)
    plt.savefig(fig_path)
    plt.close()
    plt.plot(reg2)
    fig_path = "{}/regret2_original.png".format(path)
    plt.savefig(fig_path)

    reg = {'Reg1':reg1, 'Reg2' : reg2}

    regret_path = "{}/regret.json".format(path)
    save_dictionary(reg, regret_path)


    print('------------- END of ALGORITHM  ----- time', training_time)










    #
    #
    # # testing rounds ! dont do too small!
    #
    # print("---- EVALUATION -----")
    # dataset_test = B.get_new_context_set(TT)
    # # print('dataset_test', dataset_test)
    # a_test = dataset_test.loc[:, 'sensitive_features']
    # y_test = dataset_test.loc[:, 'label']
    # xa_test = dataset_test.drop(columns=['sensitive_features', 'label', 'l0', 'l1'])
    #
    # dec_prob = best_pi.predict(xa_test)
    # scores = pd.Series(dec_prob[:, 0], name="scores_expgrad_XA").astype(int)
    #
    # acc, mean_pred, parity, FPR, TPR, EO, util  = statistics1.get_stats(y_test, scores, a_test)
    # statistics1.save_stats(acc, mean_pred, parity, FPR, TPR, EO, util, scores)
    #
    #
    # print("------------- COMPARISON without phase 2 -----")
    #
    # _y = dataset1.loc[:, 'label']
    # XA = pd.DataFrame(dataset1.drop(columns=['sensitive_features', 'label', 'l0', 'l1']))
    # A = pd.Series(dataset1.loc[:, 'sensitive_features'], name='sensitive_features')
    # L = pd.DataFrame(dataset1.loc[:, ['l0', 'l1']])
    #
    #
    # expgrad_XA = ExponentiatedGradient(
    #     dataset1,
    #     LogisticRegression(solver='liblinear', fit_intercept=True),
    #     constraints=fairness,
    #     eps=eps,
    #     nu=nu)
    #
    # expgrad_XA.fit(
    #     XA,
    #     L,
    #     sensitive_features=A)
    #
    # dec_prob = expgrad_XA.predict(xa_test)
    # scores = pd.Series(dec_prob[:, 0], name="scores_expgrad_XA").astype(int)
    #
    # acc, mean_pred, parity, FPR, TPR, EO, util = statistics2.get_stats(y_test, scores, a_test)
    # statistics2.save_stats(acc, mean_pred, parity, FPR, TPR, EO, util, scores)
