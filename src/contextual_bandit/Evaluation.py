import pandas as pd
import numpy as np
from IPython.display import display
import json
import inspect
import numbers
from src.evaluation.training_evaluation import Statistics
_L0 = "l0"
_L1 = "l1"
from src.fairlearn.metrics import mean_prediction_group_summary, accuracy_score_group_summary, \
    equalized_odds_difference, demographic_parity_difference, demographic_parity_ratio, \
    equalized_odds_ratio, true_positive_rate_difference, true_positive_rate_ratio, \
    false_positive_rate_difference, false_positive_rate_ratio, utility
# import matplotlib.pyplot as plt
from data.uncalibrated_score import UncalibratedScore

class Evaluation(object):
    def __init__(self):
        # self.acc_dict ={'overall':{}, 0:{}, 1:{}}
        # self.mean_pred_dict = {'overall':{}, 0:{}, 1:{}}
        self.DP_list = []
        self.TPR_list = []
        self.FPR_list = []
        self.EO_list = []
        self.acc_list_overall = []
        self.acc_list_0 = []
        self.acc_list_1 = []
        self.mean_pred_overall_list = []
        self.mean_pred_0_list = []
        self.mean_pred_1_list = []
        self.util_list = []

        self.stats_list = []

    def evaluate(self, pi):
    # print('--- validation oracle policy returned --- ')

        # get test set

        # shifts == True (DP), shifts = False (EO, TRP)

        fraction_protected = 0.5
        n_test = 5000
        shifts = False
        distribution = UncalibratedScore(fraction_protected)
        x_test, a_test, y_test = distribution.sample_test_dataset(n_test, shifts)
        X_test = pd.DataFrame(x_test.squeeze())
        y_test = pd.Series(y_test.squeeze(), name='label')
        A_test = pd.Series(a_test.squeeze(), name='sensitive_features_X')
        XA_test = pd.concat([X_test, A_test == 1], axis=1).astype(float)
        A_test = A_test.rename('sensitive_features')


        # get prediction
        scores = pd.Series(pi.predict(XA_test), name="scores_expgrad_XA")

        # get statistics
        #eval_stats = Statistics()

        acc, mean_pred, parity, FPR, TPR, EO, util = self.get_stats(y_test, scores, A_test)

        self.save_stats(acc, mean_pred, parity, FPR, TPR, EO, util)


    def get_stats(self, y_test, scores, A_test):

        # -------- my statistics -----------
        def summary_as_df(name, summary):
            a = pd.Series(summary.by_group)
            a['overall'] = summary.overall
            return pd.DataFrame({name: a})

        acc = summary_as_df(
            "accuracy_XA",
            accuracy_score_group_summary(y_test, scores, sensitive_features=A_test))

        mean_pred = summary_as_df(
            "acceptance_rate_XA",
            mean_prediction_group_summary(y_test, scores, sensitive_features=A_test))

        #to do: implement utility
        util = utility(y_test, scores)

        parity = demographic_parity_difference(y_test, scores, sensitive_features=A_test)
        ratio_parity = demographic_parity_ratio(y_test, scores, sensitive_features=A_test)

        TPR = true_positive_rate_difference(y_test, scores, sensitive_features=A_test)
        ratio_TPR = true_positive_rate_ratio(y_test, scores, sensitive_features=A_test)

        FPR = false_positive_rate_difference(y_test, scores, sensitive_features=A_test)
        ratio_FPR = false_positive_rate_ratio(y_test, scores, sensitive_features=A_test)

        EO = equalized_odds_difference(y_test, scores, sensitive_features=A_test)
        ratio_EO = equalized_odds_ratio(y_test, scores, sensitive_features=A_test)

        print("--- EVALUATION  ---")
        classifier_summary = pd.concat([acc, mean_pred], axis=1)
        display(classifier_summary)

        print("DP = ", parity)
        print("DP_ratio = ", ratio_parity)
        print("TPR = ", TPR)
        print("TPR_ratio = ", ratio_TPR)
        print("FPR = ", FPR)
        print("FPR_ratio = ", ratio_FPR)
        print("EO = ", EO)
        print("EO_ratio = ", ratio_EO)

        return acc, mean_pred, parity, FPR, TPR, EO, util

    def save_stats(self, acc, mean_pred, parity, FPR, TPR, EO, util):
        self.acc_list_overall.append(float(acc.loc['overall']))
        self.acc_list_0.append(float(acc.loc[0]))
        self.acc_list_1.append(float(acc.loc[1]))
        self.mean_pred_overall_list.append(float(mean_pred.loc['overall']))
        self.mean_pred_0_list.append(float(mean_pred.loc[0]))
        self.mean_pred_1_list.append(float(mean_pred.loc[1]))
        self.DP_list.append(parity)
        self.FPR_list.append(FPR)
        self.TPR_list.append(TPR)
        self.EO_list.append(EO)
        self.util_list.append(util)



    def save_dictionary(data, path):

        try:
            with open(path, 'w+') as file_path:
                json.dump(data, file_path)
        except Exception as e:
            print('Saving file {} failed with exception: \n {}'.format(path, str(e)))


    def load_dictionary(path):
        try:
            with open(path, 'r') as file_path:
                return json.load(file_path)
        except Exception as e:
            print('Loading file {} failed with exception: \n {}'.format(path, str(e)))
            return None


    def serialize_value(value):
        if isinstance(value, dict):
            return serialize_dictionary(value)
        elif isinstance(value, list):
            return serialize_list(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif inspect.isfunction(value):
            return value.__name__
        elif not (isinstance(value, str) or isinstance(value, numbers.Number) or isinstance(value, list) or isinstance(value, bool)):
            return type(value).__name__
        else:
            return value
    #
    #
    # def serialize_dictionary(dictionary):
    #     serialized_dict = copy.deepcopy(dictionary)
    #     for key, value in serialized_dict.items():
    #         serialized_dict[key] = serialize_value(value)
    #
    #     return serialized_dict
    #
    #
    # def serialize_list(unserialized_list):
    #     serialized_list = []
    #     for value in unserialized_list:
    #         serialized_list.append(serialize_value(value))
    #
    #     return serialized_list