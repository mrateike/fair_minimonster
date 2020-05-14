from fairlearn.reductions import ExponentiatedGradient
from fairlearn.reductions._moments.conditional_selection_rate import DemographicParity

from contextual_bandit import Simulators
from contextual_bandit.Minimonster import MiniMonster
import pandas as pd
from fairlearn.metrics import mean_prediction_group_summary, accuracy_score_group_summary, demographic_parity_difference, demographic_parity_ratio, \
    true_positive_rate_difference, true_positive_rate_ratio, false_positive_rate_difference, false_positive_rate_ratio, test_statistics
from IPython.display import display
_L0 = "l0"
_L1 = "l1"
import time
from sklearn.linear_model import LogisticRegression
from contextual_bandit.Evaluation import Evaluation
from pathlib import Path
import numpy as np

# class Runtime(object):

def play(T1, T2, TT, fairness, batch, eps, nu):


    # shifts = True (DP), shifts = False (TPR, EOdds)
    if fairness == "TPR":
        shifts = False
        fairness = TruePositiveRateDifference()
    elif fairness == 'DP':
        fairness = DemographicParity()
        shifts = True
    else:
        print('ERROR')

    # "Uncalibrated", "LawSchool"
    B = Simulators.DatasetBandit(shifts)

    dataset1 = B.get_new_context_set(T1)
    print('Hello')
    # for AMO
    # learning_alg = lambda: sklearn.linear_model.LogisticRegression(solver='liblinear', fit_intercept=True)
    # M = MiniMonster(B, learning_alg=learning_alg, fairness)
    M = MiniMonster(B, fairness, dataset1, eps, nu)

    print("------------- start fit ---------------")
    start = time.time()

    # input fairness
    l1, l2, Q, best_pi = M.fit(T2, batch)
    stop = time.time()
    training_time = np.array([stop - start])
    print('L_t', l1)

    ser_M = Evaluation.serialize_value(M)
    base_save_path = Path.cwd() / 'save'
    Path(base_save_path).mkdir(parents=True, exist_ok=True)
    timestamp = time.gmtime()
    ts_folder = time.strftime("%Y-%m-%d-%H-%M-%S", timestamp)
    base_save_path = "{}/{}".format(base_save_path, ts_folder)
    Path(base_save_path).mkdir(parents=True, exist_ok=True)
    parameter_save_path = "{}/model.json".format(base_save_path)
    Evaluation.save_dictionary(ser_M, parameter_save_path)

    # print('end fit: best losses', l2)
    # print('end fit: regret', [x1-x2 for x1,x2 in zip(l1,l2)])
    print('end fit: time', training_time)

    # testing rounds ! dont do too small!
    print("------------- END of ALGORITHM EVALUATION  -----")
    dataset_test = B.get_new_context_set(TT)
    print('dataset_test', dataset_test)
    a_test = dataset_test.loc[:, 'sensitive_features']
    y_test = dataset_test.loc[:, 'label']
    xa_test = dataset_test.drop(columns=['sensitive_features', 'label'])


    def summary_as_df(name, summary):
        a = pd.Series(summary.by_group)
        a['overall'] = summary.overall
        return pd.DataFrame({name: a})


    print("------------- END of ALGORITHM evaluate best_pi -----")
    scores_best_pi = pd.Series(best_pi.predict(xa_test), name="scores_expgrad_XA")
    print('best_pi decisions', scores_best_pi.tolist())
    auc_best_pi = summary_as_df(
        "accuracy_XA",
        accuracy_score_group_summary(y_test, scores_best_pi, sensitive_features=a_test))
    mean_pred_best_pi = summary_as_df(
        "acceptance_rate_XA",
        mean_prediction_group_summary(y_test, scores_best_pi, sensitive_features=a_test))
    # group_summary_best_pi = summary_as_df(
    #     "acceptance_rate_XA",
    #     group_summary(y_test, scores_best_pi, sensitive_features=a_test))


    test_statistics(y_test, a_test)

    parity_best_pi = demographic_parity_difference(y_test, scores_best_pi, sensitive_features=a_test)
    ratio_parity_best_pi = demographic_parity_ratio(y_test, scores_best_pi, sensitive_features=a_test)
    TPR_best_pi = true_positive_rate_difference(y_test, scores_best_pi, sensitive_features=a_test)
    ratio_TPR_best_pi = true_positive_rate_ratio(y_test, scores_best_pi, sensitive_features=a_test)
    FPR_best_pi = false_positive_rate_difference(y_test, scores_best_pi, sensitive_features=a_test)
    ratio_FPR_best_pi = false_positive_rate_ratio(y_test, scores_best_pi, sensitive_features=a_test)
    # equalOdds_expGrad = equalized_odds_difference(y_test, scores_expgrad_XA, sensitive_features=a_test)
    # ratio_equalOdds_expGrad = equalized_odds_ratio(y_test, scores_expgrad_XA, sensitive_features=a_test)

    classifier_summary_best_pi = pd.concat([auc_best_pi, mean_pred_best_pi], axis=1)
    display(classifier_summary_best_pi)

    print("DP = ", parity_best_pi)
    print("DP_ratio = ", ratio_parity_best_pi)
    print("TPR = ", TPR_best_pi)
    print("TPR_ratio = ", ratio_TPR_best_pi)
    print("FPR = ", FPR_best_pi)
    print("FPR_ratio = ", ratio_FPR_best_pi)
    # print("EO = ", equalOdds_expGrad)
    # print("EO_ratio = ", ratio_equalOdds_expGrad)

    print("------------- COMPARISON without phase 2 -----")

    _y = dataset1.loc[:, 'label']
    XA = pd.DataFrame(dataset1.iloc[:, 0:2])
    A = pd.Series(dataset1.loc[:, 'sensitive_features'], name='sensitive_features')
    L = pd.DataFrame(columns=['l0', 'l1'])

    for index, value in _y.items():
        if value == 0:
            L.at[index, _L0] = 0
            L.at[index, _L1] = 1
        elif value == 1:
            L.at[index, _L0] = 1
            L.at[index, _L1] = 0
        else:
            print('ERROR')

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

    scores_expgrad_XA = pd.Series(expgrad_XA.predict(xa_test), name="scores_expgrad_XA")
    print('best_pi decisions', scores_expgrad_XA.tolist())
    auc_expGrad = summary_as_df(
        "accuracy_XA",
        accuracy_score_group_summary(y_test, scores_expgrad_XA, sensitive_features=a_test))
    mean_pred_expGrad = summary_as_df(
        "acceptance_rate_XA",
        mean_prediction_group_summary(y_test, scores_expgrad_XA, sensitive_features=a_test))

    parity_expGrad = demographic_parity_difference(y_test, scores_expgrad_XA, sensitive_features=a_test)
    ratio_parity_expGrad = demographic_parity_ratio(y_test, scores_expgrad_XA, sensitive_features=a_test)
    TPR_expGrad = true_positive_rate_difference(y_test, scores_expgrad_XA, sensitive_features=a_test)
    ratio_TPR_expGrad = true_positive_rate_ratio(y_test, scores_expgrad_XA, sensitive_features=a_test)
    FPR_expGrad = false_positive_rate_difference(y_test, scores_expgrad_XA, sensitive_features=a_test)
    ratio_FPR_expGrad = false_positive_rate_ratio(y_test, scores_expgrad_XA, sensitive_features=a_test)
    # equalOdds_expGrad = equalized_odds_difference(y_test, scores_expgrad_XA, sensitive_features=a_test)
    # ratio_equalOdds_expGrad = equalized_odds_ratio(y_test, scores_expgrad_XA, sensitive_features=a_test)

    classifier_summary = pd.concat([auc_expGrad, mean_pred_expGrad], axis=1)
    display(classifier_summary)

    print("DP = ", parity_expGrad)
    print("DP_ratio = ", ratio_parity_expGrad)
    print("TPR = ", TPR_expGrad)
    print("TPR_ratio = ", ratio_TPR_expGrad)
    print("FPR = ", FPR_expGrad)
    print("FPR_ratio = ", ratio_FPR_expGrad)
    # print("EO = ", equalOdds_expGrad)
    # print("EO_ratio = ", ratio_equalOdds_expGrad)