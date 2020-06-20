import os
import sys

root_path = os.path.abspath(os.path.join(".."))
if root_path not in sys.path:
    sys.path.append(root_path)
from .._exponentiated_gradient.exponentiated_gradient  import ExponentiatedGradient
import numpy as np
import pandas as pd
import matplotlib as mpl
#TkAgg
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt


from sklearn.linear_model import LogisticRegression
from src.fairlearn.metrics import mean_prediction_group_summary, accuracy_score_group_summary, equalized_odds_difference, demographic_parity_difference, demographic_parity_ratio, equalized_odds_ratio, true_positive_rate_difference, true_positive_rate_ratio, false_positive_rate_difference, false_positive_rate_ratio
# import matplotlib.pyplot as plt


import random

# Load the data using the tempeh package
# from tempeh.configurations import datasets
# dataset = datasets['lawschool_passbar']()
#
# X_train, X_test = dataset.get_X(format=pd.DataFrame)
# y_train, y_test = dataset.get_y(format=pd.Series)
# A_train, A_test = dataset.get_sensitive_features(name='race', format=pd.Series)

def play (dataset1, dataset2, fairness, eps, nu, statistics, p):

    y2_train = dataset2.loc[:, 'label']

    i=0
    for index in dataset2.index:
        y_true =y2_train.tolist()[i]
        i+=1

        right = int(random.uniform(0,1)>p)


        # -----  bandit min Loss version 2 -------
        if right == 0:
            # if we are not deciding right
            if y_true == 0:
                #dec == 1
                dataset2.at[index, 'l0'] = 0
                dataset2.at[index, 'l1'] = 1/p
            elif y_true == 1:
                # dec == 0
                dataset2.at[index, 'l0'] = 0.5/p
                dataset2.at[index, 'l1'] = 0
            else:
                print('run_expg ERROR')
        else:
            if y_true == 0:
                # dec == 0
                dataset2.at[index, 'l0'] = 0.5/(1-p)
                dataset2.at[index, 'l1'] = 0


            elif y_true == 1:
                # dec == 1
                dataset2.at[index, 'l0'] = 0
                dataset2.at[index, 'l1'] = 0/(1-p)

            else:
                print('run_expg ERROR')
        # index += 1

        # # ---- supervised learning -----
        # if y_true == 0:
        #     l_hat.at[index, 'l0'] = 0
        #     l_hat.at[index, 'l1'] = 1
        # else:
        #     l_hat.at[index, 'l0'] = 1
        #     l_hat.at[index, 'l1'] = 0
        # index += 1

        # -----  bandit min Loss -------
        # right = int(random.uniform(0,1)>0.3)
        # if right == 0:
        #     if y_true == 0:
        #         #dec == 1
        #         l_hat.at[index, 'l0'] = 0/0.7
        #         l_hat.at[index, 'l1'] = 1/0.3
        #     else:
        #         # dec == 0
        #         l_hat.at[index, 'l0'] = 0.5/0.3
        #         l_hat.at[index, 'l1'] = 0.5/0.7
        # else:
        #     if y_true == 0:
        #         # dec == 0
        #         l_hat.at[index, 'l0'] = 0.5/0.7
        #         l_hat.at[index, 'l1'] = 0.5/0.3
        #     else:
        #         # dec == 1
        #         l_hat.at[index, 'l0'] = 1/0.3
        #         l_hat.at[index, 'l1'] = 0/0.7
        # index += 1


        # -----  bandit max Var -------
        # right = int(random.uniform(0, 1) > 0.3)
        # mu =0
        # if right == 0:
        #     if y_true == 0:
        #         # dec == 1
        #         l_hat.at[index, 'l0'] = (0-mu) / 0.7
        #         # l_hat.at[index, 'l1'] = (1-mu) / 0.3
        #     else:
        #         # dec == 0
        #         l_hat.at[index, 'l0'] = (0.5-mu) / 0.3
        #         l_hat.at[index, 'l1'] = (0.5-mu) / 0.7
        # else:
        #     if y_true == 0:
        #         # dec == 0
        #         l_hat.at[index, 'l0'] = (0.5-mu) / 0.7
        #         l_hat.at[index, 'l1'] = (0.5-mu) / 0.3
        #     else:
        #         # dec == 1
        #         l_hat.at[index, 'l0'] = (1-mu) / 0.3
        #         l_hat.at[index, 'l1'] = (0-mu) / 0.7
        # index += 1

        # #-----  bandit max Var version 2 -------
        # right = int(random.uniform(0, 1) > 0.3)
        # mu = 0.1
        # if right == 0:
        #     if y_true == 0:
        #         # dec == 1
        #         l_hat.at[index, 'l0'] = (1) - mu / 0.7
        #         l_hat.at[index, 'l1'] = (1-mu) / 0.3
        #     else:
        #         # dec == 0
        #         l_hat.at[index, 'l0'] = (0.5-mu) / 0.3
        #         l_hat.at[index, 'l1'] = (0.5)-mu / 0.7
        # else:
        #     if y_true == 0:
        #         # dec == 0
        #         l_hat.at[index, 'l0'] = (0.5-mu) / 0.7
        #         l_hat.at[index, 'l1'] = (0.5)-mu / 0.3
        #     else:
        #         # dec == 1
        #         l_hat.at[index, 'l0'] = (1)-mu / 0.3
        #         l_hat.at[index, 'l1'] = (1-mu) / 0.7
        # index += 1


    A2_train = dataset2.loc[:, 'sensitive_features']
    l_IPS = dataset2.filter(items=['l0', 'l1'])
    XA2_train = dataset2.filter(items=['features', 'sensitive_features'])


    expgrad_XA = ExponentiatedGradient(
        dataset1,
        LogisticRegression(solver='liblinear', fit_intercept=True),
        constraints=fairness,
        eps=eps,
        nu=nu)

    expgrad_XA.fit(
        XA2_train,
        l_IPS,
        sensitive_features=A2_train)

    statistics.evaluate(expgrad_XA)

    # #---- Evaluation of log regressor -----
    # logReg_predictor = LogisticRegression(solver='liblinear', fit_intercept=True)
    # logReg_predictor.fit(XA2_train, y2_train)
