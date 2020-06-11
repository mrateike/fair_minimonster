from src.fairlearn.reductions._exponentiated_gradient.exponentiated_gradient import ExponentiatedGradient
from src.fairlearn.reductions._moments.conditional_selection_rate import DemographicParity

from src.contextual_bandit import Simulators
from src.contextual_bandit.Minimonster import MiniMonster
import pandas as pd

_L0 = "l0"
_L1 = "l1"
import time
from sklearn.linear_model import LogisticRegression
from src.evaluation.Evaluation import Evaluation
import numpy as np

# class Runtime(object):

def play(T1, T2, TT, fairness, batch, batchsize, eps, nu, dataset):

    statistics1 = Evaluation(TT)
    statistics2 = Evaluation(TT)
    # # shifts = True (DP), shifts = False (TPR, EOdds)
    if fairness == "TPR":
        fairness = TruePositiveRateDifference()
    elif fairness == 'DP':
        fairness = DemographicParity()
    # else:
    #     print('ERROR')

    # "Uncalibrated", "FICO", "Adult", 'COMPAS'
    # dataset = 'Uncalibrated'
    B = Simulators.DatasetBandit(dataset)

    # returns: XA, L(l0, l1), A, Y
    dataset1 = B.get_new_context_set(T1)

    M = MiniMonster(B, fairness, dataset1, eps, nu, TT)

    print("------------- start fit ---------------")
    start = time.time()

    # input fairness
    l1, l2, Q, best_pi = M.fit(T2, batch, batchsize)
    stop = time.time()
    training_time = np.array([stop - start])
    # print('L_t', l1)


    # save model


    print('------------- END of ALGORITHM  ----- time', training_time)

    # testing rounds ! dont do too small!

    print("---- EVALUATION -----")
    dataset_test = B.get_new_context_set(TT)
    # print('dataset_test', dataset_test)
    a_test = dataset_test.loc[:, 'sensitive_features']
    y_test = dataset_test.loc[:, 'label']
    xa_test = dataset_test.drop(columns=['sensitive_features', 'label', 'l0', 'l1'])

    dec_prob = best_pi.predict(xa_test)
    scores = pd.Series(dec_prob[:, 0], name="scores_expgrad_XA").astype(int)

    acc, mean_pred, parity, FPR, TPR, EO, util  = statistics1.get_stats(y_test, scores, a_test)
    statistics1.save_stats(acc, mean_pred, parity, FPR, TPR, EO, util, scores)

    # def summary_as_df(name, summary):
    #     a = pd.Series(summary.by_group)
    #     a['overall'] = summary.overall
    #     return pd.DataFrame({name: a})
    #
    #
    # print("------------- END of ALGORITHM evaluate best_pi -----")
    # scores_best_pi = pd.Series(best_pi.predict(xa_test), name="scores_expgrad_XA")
    # print('best_pi decisions', scores_best_pi.tolist())
    # auc_best_pi = summary_as_df(
    #     "accuracy_XA",
    #     accuracy_score_group_summary(y_test, scores_best_pi, sensitive_features=a_test))
    # mean_pred_best_pi = summary_as_df(
    #     "acceptance_rate_XA",
    #     mean_prediction_group_summary(y_test, scores_best_pi, sensitive_features=a_test))
    # # group_summary_best_pi = summary_as_df(
    # #     "acceptance_rate_XA",
    # #     group_summary(y_test, scores_best_pi, sensitive_features=a_test))
    #
    #
    # test_statistics(y_test, a_test)
    #
    # parity_best_pi = demographic_parity_difference(y_test, scores_best_pi, sensitive_features=a_test)
    # ratio_parity_best_pi = demographic_parity_ratio(y_test, scores_best_pi, sensitive_features=a_test)
    # TPR_best_pi = true_positive_rate_difference(y_test, scores_best_pi, sensitive_features=a_test)
    # ratio_TPR_best_pi = true_positive_rate_ratio(y_test, scores_best_pi, sensitive_features=a_test)
    # FPR_best_pi = false_positive_rate_difference(y_test, scores_best_pi, sensitive_features=a_test)
    # ratio_FPR_best_pi = false_positive_rate_ratio(y_test, scores_best_pi, sensitive_features=a_test)
    # # equalOdds_expGrad = equalized_odds_difference(y_test, scores_expgrad_XA, sensitive_features=a_test)
    # # ratio_equalOdds_expGrad = equalized_odds_ratio(y_test, scores_expgrad_XA, sensitive_features=a_test)
    #
    # classifier_summary_best_pi = pd.concat([auc_best_pi, mean_pred_best_pi], axis=1)
    # display(classifier_summary_best_pi)
    #
    # print("DP = ", parity_best_pi)
    # print("DP_ratio = ", ratio_parity_best_pi)
    # print("TPR = ", TPR_best_pi)
    # print("TPR_ratio = ", ratio_TPR_best_pi)
    # print("FPR = ", FPR_best_pi)
    # print("FPR_ratio = ", ratio_FPR_best_pi)
    # # print("EO = ", equalOdds_expGrad)
    # # print("EO_ratio = ", ratio_equalOdds_expGrad)

    print("------------- COMPARISON without phase 2 -----")

    _y = dataset1.loc[:, 'label']
    XA = pd.DataFrame(dataset1.drop(columns=['sensitive_features', 'label', 'l0', 'l1']))
    A = pd.Series(dataset1.loc[:, 'sensitive_features'], name='sensitive_features')
    L = pd.DataFrame(dataset1.loc[:, ['l0', 'l1']])


    expgrad_XA = ExponentiatedGradient(
        dataset1,
        LogisticRegression(solver='liblinear', fit_intercept=True),
        constraints=fairness,
        eps=eps,
        nu=nu)

    expgrad_XA.fit(
        XA,
        L,
        sensitive_features=A)

    dec_prob = expgrad_XA.predict(xa_test)
    scores = pd.Series(dec_prob[:, 0], name="scores_expgrad_XA").astype(int)

    acc, mean_pred, parity, FPR, TPR, EO, util = statistics2.get_stats(y_test, scores, a_test)
    statistics2.save_stats(acc, mean_pred, parity, FPR, TPR, EO, util, scores)
