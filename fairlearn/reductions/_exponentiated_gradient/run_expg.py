from fairlearn.reductions import ExponentiatedGradient
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from fairlearn.metrics import mean_prediction_group_summary, accuracy_score_group_summary, equalized_odds_difference, demographic_parity_difference, demographic_parity_ratio, equalized_odds_ratio, true_positive_rate_difference, true_positive_rate_ratio, false_positive_rate_difference, false_positive_rate_ratio
# import matplotlib.pyplot as plt
from data.uncalibrated_score import UncalibratedScore
# Load the data using the tempeh package
# from tempeh.configurations import datasets
# dataset = datasets['lawschool_passbar']()
#
# X_train, X_test = dataset.get_X(format=pd.DataFrame)
# y_train, y_test = dataset.get_y(format=pd.Series)
# A_train, A_test = dataset.get_sensitive_features(name='race', format=pd.Series)
class RunExpg:
    def play (n1_train, n2_train, n_test, shifts, fairness):

        fraction_protected = 0.5

        distribution = UncalibratedScore(fraction_protected)
        x1_train, a1_train, y1_train = distribution.sample_train_dataset(n1_train, shifts)
        x2_train, a2_train, y2_train = distribution.sample_train_dataset(n2_train, shifts)
        x_test, a_test, y_test = distribution.sample_test_dataset(n_test, shifts)

        y2_train = pd.Series((y2_train.squeeze()))

        l_hat = pd.DataFrame(np.ones((x2_train.shape[0], 2)), index=range(len(y2_train)), columns=['l0', 'l1'])

        index = 0
        for item in y2_train.tolist():
                if item == 0:
                    #----- perfect bandit -------
                    # l_hat.at[index, 'l0'] = 0.5/0.9
                    # l_hat.at[index, 'l1'] = 0.5/0.1
                    #---- supervised learning -----
                    l_hat.at[index, 'l0'] = 0
                    l_hat.at[index, 'l1'] = 1
                else:
                    #------ perfect bandit
                    # l_hat.at[index,'l0'] = 1/0.9
                    # l_hat.at[index,'l1'] = 0/0.1
                    # #---- supervised learning
                    l_hat.at[index, 'l0'] = 1
                    l_hat.at[index, 'l1'] = 0
                index +=1
        l_IPS  = l_hat.set_index(y2_train.index)
        # display(l_IPS)

        X1_train = pd.DataFrame(x1_train.squeeze())
        X2_train = pd.DataFrame(x2_train.squeeze())
        X_test = pd.DataFrame(x_test.squeeze())

        y1_train = pd.Series(y1_train.squeeze(), name='label')
        y2_train = pd.Series(y2_train.squeeze(), name='label')
        y_test = pd.Series(y_test.squeeze(), name='label')

        # need a different name for classifier family sens_equal and sens_flip
        A1_train = pd.Series(a1_train.squeeze(), name='sensitive_features_X')
        A2_train = pd.Series(a2_train.squeeze(), name='sensitive_features_X')
        A_test = pd.Series(a_test.squeeze(), name='sensitive_features_X')

        # black, women == 1
        XA1_train = pd.concat([X1_train, A1_train==1], axis=1).astype(float)
        XA2_train = pd.concat([X2_train, A2_train==1], axis=1).astype(float)
        XA_test = pd.concat([X_test, A_test==1], axis=1).astype(float)

        A1_train = A1_train.rename('sensitive_features')
        A2_train = A2_train.rename('sensitive_features')
        A_test = A_test.rename('sensitive_features')



        # Combine all training data into a single data frame and glance at a few rows
        dataset_phase1X = pd.concat([X1_train, y1_train, A1_train], axis=1)
        dataset_phase1XA = pd.concat([XA1_train, y1_train, A1_train], axis=1)



        logReg_predictor = LogisticRegression(solver='liblinear', fit_intercept=True)
        logReg_predictor.fit(XA2_train, y2_train)

        # a convenience function that transforms the result of a group metric call into a data frame
        def summary_as_df(name, summary):
            a = pd.Series(summary.by_group)
            a['overall'] = summary.overall
            return pd.DataFrame({name: a})

        scores_logReg = pd.Series(logReg_predictor.predict(XA_test), name="score_unmitigated")

        auc_logReg = summary_as_df(
             "accuracy_unmitigated_predictor",
             accuracy_score_group_summary(y_test, scores_logReg, sensitive_features=A_test))
        sel_logReg = summary_as_df(
            "selection_unmitigated_predictor",
            mean_prediction_group_summary(y_test, scores_logReg, sensitive_features=A_test))

        classifier_summary = pd.concat([auc_logReg, sel_logReg], axis=1)
        # classifier_summary.loc['disparity']=(classifier_summary.iloc[1]-classifier_summary.iloc[0]).abs()
        # classifier_summary.loc['disparity', classifier_summary.columns.str.startswith('acc')]='-'
        # display(classifier_summary)


        parity_logReg = demographic_parity_difference(y_test, scores_logReg, sensitive_features=A_test)
        ratio_parity_logReg = demographic_parity_ratio(y_test, scores_logReg, sensitive_features=A_test)
        TPR_logReg = true_positive_rate_difference(y_test, scores_logReg, sensitive_features=A_test)
        ratio_TPR_logReg = true_positive_rate_ratio(y_test, scores_logReg, sensitive_features=A_test)
        FPR_logReg = false_positive_rate_difference(y_test, scores_logReg, sensitive_features=A_test)
        ratio_FPR_logReg = false_positive_rate_ratio(y_test, scores_logReg, sensitive_features=A_test)
        equalOdds_logReg = equalized_odds_difference(y_test, scores_logReg, sensitive_features=A_test)
        ratio_equalOdds_logReg = equalized_odds_ratio(y_test, scores_logReg, sensitive_features=A_test)

        # print("DP = ", parity_logReg)
        # print("DP_ratio = ", ratio_parity_logReg)
        # print("TPR = ", TPR_logReg)
        # print("TPR_ratio = ", ratio_TPR_logReg)
        # print("FPR = ", FPR_logReg)
        # print("FPR_ratio = ", ratio_FPR_logReg)
        # print("EO = ", equalOdds_logReg)
        # print("EO_ratio = ", ratio_equalOdds_logReg)

        # balanced_index_pass0 = y_train[y_train==0].index
        # balanced_index_pass1 = y_train[y_train==1].sample(n=balanced_index_pass0.size, random_state=0).index
        # balanced_index = balanced_index_pass0.union(balanced_index_pass1)

        # print('dataset_phase1X',dataset_phase1X)

        expgrad_XA = ExponentiatedGradient(
            dataset_phase1XA,
            LogisticRegression(solver='liblinear', fit_intercept=True),
            constraints=fairness,
            eps=0.01,
            nu=1e-6)

        expgrad_XA.fit(
            XA2_train,
            l_IPS,
            sensitive_features=A2_train)

        scores_expgrad_XA = pd.Series(expgrad_XA.predict(XA_test), name="scores_expgrad_XA")


        auc_expGrad = summary_as_df(
            "accuracy_XA",
            accuracy_score_group_summary(y_test, scores_expgrad_XA, sensitive_features=A_test))
        mean_pred_expGrad = summary_as_df(
            "acceptance_rate_XA",
            mean_prediction_group_summary(y_test, scores_expgrad_XA, sensitive_features=A_test))


        parity_expGrad = demographic_parity_difference(y_test, scores_expgrad_XA, sensitive_features=A_test)
        ratio_parity_expGrad = demographic_parity_ratio(y_test, scores_expgrad_XA, sensitive_features=A_test)
        TPR_expGrad = true_positive_rate_difference(y_test, scores_expgrad_XA, sensitive_features=A_test)
        ratio_TPR_expGrad = true_positive_rate_ratio(y_test, scores_expgrad_XA, sensitive_features=A_test)
        FPR_expGrad = false_positive_rate_difference(y_test, scores_expgrad_XA, sensitive_features=A_test)
        ratio_FPR_expGrad = false_positive_rate_ratio(y_test, scores_expgrad_XA, sensitive_features=A_test)
        equalOdds_expGrad = equalized_odds_difference(y_test, scores_expgrad_XA, sensitive_features=A_test)
        ratio_equalOdds_expGrad = equalized_odds_ratio(y_test, scores_expgrad_XA, sensitive_features=A_test)

        classifier_summary = pd.concat([auc_expGrad, mean_pred_expGrad], axis=1)
        # classifier_summary.loc['disparity']=(classifier_summary.iloc[1]-classifier_summary.iloc[0]).abs()
        # classifier_summary.loc['disparity', classifier_summary.columns.str.startswith('auc')]='-'
        # display(classifier_summary)

        # print("DP = ", parity_expGrad)
        # print("DP_ratio = ", ratio_parity_expGrad)
        # print("TPR = ", TPR_expGrad)
        # print("TPR_ratio = ", ratio_TPR_expGrad)
        # print("FPR = ", FPR_expGrad)
        # print("FPR_ratio = ", ratio_FPR_expGrad)
        # print("EO = ", equalOdds_expGrad)
        # print("EO_ratio = ", ratio_equalOdds_expGrad)



        improved_DP = parity_logReg - parity_expGrad
        improved_ratio_parity = ratio_parity_expGrad - ratio_parity_logReg
        improved_TPR = TPR_logReg - TPR_expGrad
        improved_ratio_TPR = ratio_TPR_expGrad - ratio_TPR_logReg
        improved_FPR = FPR_logReg - FPR_expGrad
        improved_ratio_FPR = ratio_FPR_expGrad - ratio_FPR_logReg
        improved_equalOdds = equalOdds_logReg - equalOdds_expGrad
        improved_ratio_equalOdds = ratio_equalOdds_expGrad - ratio_equalOdds_logReg
        improved_acc = (auc_expGrad.loc['overall'].values - auc_logReg.loc['overall'].values).squeeze()

        # print("------- IMPROVEMENT of Fairlearn --------------")
        # print("DP = ", improved_DP)
        # print("DP_ratio = ", improved_ratio_parity)
        # print("TPR = ", improved_TPR)
        # print("TPR_ratio = ", improved_ratio_TPR)
        # print("FPR = ", improved_FPR)
        # print("FPR_ratio = ", improved_ratio_FPR)
        # print("EO = ", improved_equalOdds)
        # print("EO_ratio = ", improved_ratio_equalOdds)
        # print("------- Loss of Fairlearn --------------")
        # print("Accuracy = ", improved_acc)

        return improved_acc, improved_DP, improved_TPR, improved_FPR, improved_equalOdds