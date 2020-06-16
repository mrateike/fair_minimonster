import time
from pathlib import Path
import matplotlib as mpl
# mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.contextual_bandit import Argmin
from src.evaluation.Evaluation import Evaluation, save_and_plot_results, my_plot, my_plot2
from src.evaluation.training_evaluation import Statistics
from data.util import save_dictionary
from src.evaluation.training_evaluation import UTILITY
import math


import os

my_path = os.path.abspath(__file__)  # Figures out the absolute path for you in case your working directory moves around.


class MiniMonster(object):
    """
    Implementation of MiniMonster with a scikit_learn learning algorithm as the AMO.
    """

    def __init__(self, B, fairness, dataset1, eps, nu, TT, test_seed, dataset2, path, mu):
        self.B = B
        # XA, L, A, Y
        self.dataset1 = dataset1
        self.dataset2 = dataset2.drop(['l0', 'l1'], axis=1)
        self.history = dataset1

        loss_path = "{}/loss_policies_results".format(path)
        Path(loss_path).mkdir(parents=True, exist_ok=True)
        loss_path = "{}/loss_".format(loss_path)

        var_path = "{}/var_policies_results".format(path)
        Path(var_path).mkdir(parents=True, exist_ok=True)
        var_path = "{}/var_".format(var_path)

        dec_path = "{}/bandit_decions_results".format(path)
        Path(dec_path).mkdir(parents=True, exist_ok=True)
        dec_path = "{}/dec_".format(dec_path)

        self.regret_path = "{}/bandit_regret_results".format(path)
        Path(self.regret_path).mkdir(parents=True, exist_ok=True)

        self.statistics_loss = Evaluation(TT, test_seed, loss_path, B)
        self.statistics_var = Evaluation(TT, test_seed, var_path, B)
        self.statistics_decisions = Evaluation(TT, test_seed, dec_path, B)


        self.mu = mu

        # XA, L, A (supervised)

        # print('history', self.history)
        self.num_amo_calls = 0
        self.mu = 0.1
        self.fairness = fairness

        self.eps = eps
        self.nu = nu

        self.DP_dict = {}
        self.TPR_dict = {}
        self.FPR_dict = {}
        self.EO_dict = {}
        self.auc_dict = {}
        self.mean_pred_dict = {}




    def fit(self, T2, T1, batch, batchsize):
        real_loss = []
        best_loss_T = []
        best_loss_t = []




        XA = pd.DataFrame()
        A = pd.Series(name='sensitive_features')
        L = pd.DataFrame(columns=['l0', 'l1'])
        dataset2_collected = pd.DataFrame()

        # for _solve_Opt
        psi = 4*(math.e -2)*np.log(T2+T1)


        # predictions = {}
        Q = []
        # best_pi = None



        print('--- EVALUATION --  first best policy on phase 1 data')
        # print('self.dataset1', self.dataset1)
        best_pi  = Argmin.argmin(self.eps, self.nu, self.fairness, self.dataset1)
        self.statistics_loss.evaluate(best_pi.model)

        for i in self.dataset1['l1'].values:
            real_loss.append(i)
        for i in range(0, self.dataset1.shape[0]):
            individual1 = self.dataset1.iloc[i]
            xa1 = individual1.loc[['features', 'sensitive_features_X']].to_frame().T
            d1 = best_pi.predict(xa1).squeeze()[0]
            if d1 == 1:
                l1 = individual1.loc['l1']
            elif d1 == 0:
                l1 = individual1.loc['l0']

            best_loss_t.append(l1)

        print('len(real_loss)', len(real_loss))
        print('len(best_loss_t)', len(best_loss_t))


        # ------------- batch settings
        training_points = []
        m = 1
        if batch == "exp":
            while True:
                training_points.append(int(2 ** (m-1)))
                m += 1
                if (2 ** (m-1)) > T2:
                    break
        elif batch == 'lin':
            while True:
                batchsize = int(batchsize)
                training_points.append(int(m * batchsize))
                m += 1
                if int(m * batchsize) > T2:
                    break
        elif batch == 'warm_start':
           training_points = [3,5]
           m +=2
           while True:
               training_points.append(int(m ** 2))
               m += 1
               if int(m ** 2) > T2:
                   break

        elif batch == 'no_batch':
            # need to collect a few first such that we do not get a label problem
            m=1
            while True:
                training_points.append(int(m))
                m += 1
                if int(m) > T2:
                    break
        else:
            print('ERROR in batches')



        #----------- run loop --------------
        m=0
        t = 0

        while t < T2:
            t+= 1
            m+=1
            individual = self.dataset2.iloc[(t-1)]

            xa = individual.loc[['features', 'sensitive_features_X']].to_frame().T
            a = pd.Series(individual.loc['sensitive_features'], name='sensitive_features', index=[individual.name])

            XA = XA.append(xa)
            A = A.append(a)


            d, p = self.sample(xa, Q, best_pi, m, self.statistics_decisions)
            y = individual.loc['label']
            l = self.B.get_loss(d, y)
            real_loss.append(l)



            # IPS loss, attention: order of columns might change
            # sometimes [l1, l0], sometimes [l0, l1]
            ips_loss = pd.DataFrame()
            ips_loss.at[individual.name,str(d)] = l / p
            ips_loss.at[individual.name, str(1-d)] = 0
            ips_loss = ips_loss.rename(columns={'0':'l0','1':'l1'})


            L = L.append(ips_loss)




            # ---------- update batch ( push new ones) --------
            if t in training_points:
                print('in update')

                # real_loss.extend(losses)
                # data collected in this batch
                dataset_batch = pd.concat([XA, L, A], axis=1)

                # data collected so far in phase 2
                dataset2_collected = dataset2_collected.append(dataset_batch)

                print('--- EVALUATION -- batch update ', m, 'at time', t, 'best policy')
                # print('dataset2_collected',dataset2_collected)
                best_pi = Argmin.argmin(self.eps, self.nu, self.fairness, self.dataset1, dataset2_collected)

                db = best_pi.predict(xa).squeeze()[0]
                if db == 1:
                    lb = ips_loss.filter(['l1']).values.squeeze()
                elif db == 0:
                    lb = ips_loss.filter(['l0']).values.squeeze()
                best_loss_t.append(lb)


                self.statistics_loss.evaluate(best_pi.model)
                self.num_amo_calls += 1


                # data collected from phase 1 and phase 2 so far
                self.history = self.history.append(dataset_batch, ignore_index=True)

                # update Q
                Q = self._solve_op(self.history, T1, m, best_pi, psi)

                # clear batch containers
                XA = pd.DataFrame()
                A = pd.Series(name='sensitive_features')
                L = pd.DataFrame()
                losses = []
                # ---------- END  batch update --------

            # ---------- END  loop --------

        # self.statistics_final.evaluate(best_pi)


        # ----- Calculate regret at each step ------------

        print('len(real_loss)', len(real_loss))
        print('len(best_loss_t)', len(best_loss_t))

        regt0 = [max((real_loss[i] - best_loss_t[i]), 0) for i in range(0, len(real_loss))]
        regt0_cum = np.cumsum(regt0).tolist()
        regt = [(real_loss[i] - best_loss_t[i]) for i in range(0, len(real_loss))]
        regt_cum = np.cumsum(regt).tolist()

        regret_path = "{}/round_".format(self.regret_path)

        my_plot2(regret_path, regt, regt_cum, regt0, regt0_cum)

        # ----- Calculate regret T in hindsight ------------
        subset = self.history[['features', 'sensitive_features_X']]
        for i, xa in subset.iterrows():
            xa = xa.to_frame().T
            d = best_pi.get_decision(xa).squeeze()[0]
            if d == 1:
                l = self.history.loc[i,'l1']
            elif d == 0:
                l = self.history.loc[i, 'l0']
            else:
                print('ERROR Minimonster fit')
            best_loss_T.append(l)

        regT0 = [max((real_loss[i] - best_loss_T[i]) ,0) for i in range(0,len(real_loss))]
        regT0_cum = np.cumsum(regT0).tolist()
        RT0 = regT0_cum[-1]
        regT = [(real_loss[i] - best_loss_T[i]) for i in range(0, len(real_loss))]
        regT_cum = np.cumsum(regT).tolist()
        RT = regT_cum[-1]

        regret_path2 = "{}/finalT_".format(self.regret_path)

        my_plot2(regret_path2, regT, regT_cum, regT0, regT0_cum)

        reg_dict = {}

        reg_dict['RT'] = RT
        reg_dict['RT0'] = RT0

        reg_dict['regt'] = regt
        reg_dict['regt_cum'] = regt_cum

        reg_dict['regt0'] = regt0
        reg_dict['regt0_cum'] = regt0_cum

        reg_dict['regT'] = regT
        reg_dict['regT_cum'] = regT_cum

        reg_dict['regT0'] = regT0
        reg_dict['regT0_cum'] = regT0_cum

        regret_path = "{}/regret.json".format(self.regret_path)
        save_dictionary(reg_dict, regret_path)



















    # def predict(self, X, Q, best_pi):
    #     decisions = []
    #     curr_idx = 0
    #     prob = np.zeros(2)
    #
    #     for item in X:
    #         features = pd.DataFrame({"credit_score_feature": item[0].squeeze(), "example_sensitive_featrue": item[1].squeeze()},
    #                             index=[curr_idx])
    #         x = Context.Context(curr_idx, features)
    #
    #         for item in Q:
    #             pi = item[0]
    #             w = item[1]
    #             decis = pi.get_decision(x)
    #             prob[decis] += w
    #
    #         assert best_pi != None, 'no best policy'
    #
    #         dec = best_pi.get_decision(x)
    #         w = 1 - np.sum([z[1] for z in Q])
    #         prob[dec] += w
    #
    #         assert prob[1] >= 0 and prob[1] <= 1, 'probability needs to be [0,1]'
    #
    #         d = np.random.binomial(1, prob[1])
    #
    #         if isinstance(d, np.ndarray):
    #             d = d.squeeze()
    #         decisions.append(d)
    #         curr_idx +=1
    #         prob = np.zeros(2)
    #
    #     return decisions

    def sample(self, x, Q, best_pi, m, statistics):
        # xa = pd.DataFrame
        # Q = [Policy,float]
        # best_pi = Policy

        xa_test = statistics.XA_test
        p = np.zeros(2)
        p_t = np.zeros((xa_test.shape[0], 2))
        if Q == []:
            dec_prob = best_pi.predict(x).squeeze()
            dec = int(dec_prob[0])
            prob = dec_prob[1]
            p[dec] = prob
            p[1-dec] = 1-prob

            dec_prob_t = best_pi.predict(xa_test)
            dec_t = dec_prob_t[:, 0].astype(int)
            prob_t = dec_prob_t[:, 1]

            for i in range(0, len(dec_t)):
                dt = dec_t[i]
                pt = prob_t[i]
                p_t[i][dt] = pt
                p_t[i][1-dt] = 1-pt




        else:
            for item in Q:
                pi = item[0]
                w = item[1]

                dec_prob = pi.get_decision(x).squeeze()
                dec = int(dec_prob[0])
                prob = dec_prob[1]
                p[dec] += w*prob
                p[1-dec]+=w*(1-prob)

                dec_prob_t = best_pi.predict(xa_test)
                dec_t = dec_prob_t[:, 0].astype(int)
                prob_t = dec_prob_t[:, 1]

                for i in range(0, len(dec_t)):
                    dt = dec_t[i]
                    pt = prob_t[i]
                    p_t[i][dt] = w*pt
                    p_t[i][1 - dt] = w*(1 - pt)

                # print('p_t', p_t)

            ## Mix in leader
            bp_dec_prob = best_pi.get_decision(x).squeeze()
            bp_dec = int(bp_dec_prob[0])
            bp_prob = bp_dec_prob[1]

            bp_dec_prob_t = best_pi.get_decision(xa_test)
            # print('bp_dec_prob_t', bp_dec_prob_t)
            bp_dec_t = bp_dec_prob_t[:,0]
            bp_prob_t = bp_dec_prob_t[:,1]


            # print('Q', Q)
            bp_w = 1 - np.sum([q[1] for q in Q])

            p[bp_dec] += (bp_w*bp_prob)
            p[1-bp_dec] += (bp_w*(1-bp_prob))

            for i in range(0, len(bp_dec_t)):
                bp_dt = int(bp_dec_t[i])
                bp_pt = bp_prob_t[i]
                p_t[i][bp_dt] += (bp_w * bp_pt)
                p_t[i][(1 - bp_dt)] += (bp_w * (1 - bp_pt))



        # print('before mu p_t', p_t)
        mu = self._get_mu(m)
        p = (1 - 2 *  mu) * p + mu
        p_t = (1 - 2 * mu) * p_t + mu
        # print('after mu p_t', p_t)

        # assert p[1] >= 0 and p[1] <= 1 and p[1]+p[0] <=1, 'probability needs to be [0,1]'

        random_threshold = np.random.rand(1)
        dec = (p[1] >= random_threshold) * 1
        dec = dec.squeeze()

        for i in range(0, len(p_t)):
            random_threshold = np.random.rand(1)
            dec_t[i] = (p_t[i][1] >= random_threshold) * 1
            dec_t[i] = dec_t[i].squeeze()

        # print('dec_t', dec_t)
        # 1 / 0

        print('--- EVALUATION -- learners decision making')
        scores_test = pd.Series(dec_t)
        # print('scores_test', scores_test)
        statistics.evaluate_scores(scores_test)

        return dec, p[dec]

    def _get_mu(self, t):
        """
        Return the current value of mu_t
        """
        # a = 1.0/(4)
        # b = np.sqrt(np.log(16.0*(self.t**2)*self.B.N/self.delta)/float(4*self.t))
        a = self.mu
        b = self.mu * np.sqrt(2) / np.sqrt(t)
        c = np.min([a, b])
        return np.min([1, c])
        # return 0.1

    def _solve_op(self, H, T1, m, best_pi, psi, Q=None):

        """
        Main optimization logic for MiniMonster.
        """
        print('----- BEGIN SOLVE OP ----------')
        mu = self._get_mu(m)

        # print('H', H)
        t = H.shape[0]

        # Todo: implement warm start option
        Q = []  ## self.weights
        # ## Warm-starting Q = Q


        predictions = {}

        # H needs xa, l1, l0
        leader_loss, predictions = self.get_cum_loss(H, best_pi, predictions)


        q_losses = {}
        # only makes sense, if it is a warm start, otherwise Q is empty anyway
        # for item in Q:
        #     pi = item[0]
        #     (tmp,predictions) = self.get_cum_loss(H, pi, predictions, features=features)
        #     q_losses[pi] = tmp

        iterations = 0

        # normally loop here small iterations (because otherwise get a problem of <2 labels
        updated = True
        while updated and iterations < 20:
            iterations += 1
            updated = False

            ## First IF statement, leader_reward needs to be IPS as well as q_rewards, deleted self.B.K*(leader_rew..)
            score = np.sum([x[1] * (4 + (q_losses[x[0]] - leader_loss) / (psi * t * mu)) for x in Q])

            if score > 4:
                # print('---- First Constraint: broken ----- score ', score, '> 4')
                c = 4 / score
                Q = [(x[0], c * x[1]) for x in Q]
                updated = True
            else:
                print('')
                # print('---- First Constraint: OK ----- score', score)

            Vpi_dataset = pd.DataFrame()
            Spi_dataset = pd.DataFrame()
            Dpimin_dataset = pd.DataFrame()
            Dpinormal_dataset = pd.DataFrame()

            # sum over all dataset t (phase 1 and 2)
            # print('H.index', H.index)
            for i in H.index:

                # x = pd.DataFrame(H.loc[i, ['features','sensitive_features_X']]).transpose()
                # x = x.reset_index()
                # a = pd.Series(H.loc[i, 'sensitive_features'], name='sensitive_features')

                # Todo:2 times because later 2 times (?)
                loss = [H.loc[i, 'l0'], H.loc[i, 'l1']]



                q = np.zeros(2, dtype=np.longfloat)

                for item in Q:
                    pi = item[0]
                    w = item[1]
                    q[int(predictions[pi][i][0])] += w

                q = (1.0 - 2*mu) * q + (mu)


                v = 1 / (t * q)
                l = [(ell *psi * mu) / t for ell in loss]
                s = 1.0 / (t * (q ** 2))


                _H = H.drop(columns = ['l0', 'l1']).loc[i,:].to_frame().T.reset_index(drop=True)

                loss1 = pd.DataFrame([l - v], columns=['l0', 'l1'])
                d1 = pd.concat([_H,loss1], axis=1)
                Dpimin_dataset = Dpimin_dataset.append(d1, ignore_index=True)

                loss2 = pd.DataFrame([v - l], columns=['l0', 'l1'])
                loss2.reset_index(drop=True, inplace=True)
                d2 = pd.concat([_H,loss2], axis=1)
                Dpinormal_dataset = Dpinormal_dataset.append(d2, ignore_index=True)

                loss3 = pd.DataFrame([v], columns=['l0', 'l1'])
                loss3.reset_index(drop=True, inplace=True)
                v = pd.concat([_H,loss3], axis=1)
                Vpi_dataset = Vpi_dataset.append(v, ignore_index=True)

                loss4 = pd.DataFrame([s], columns=['l0', 'l1'])
                loss4.reset_index(drop=True, inplace=True)
                s = pd.concat([_H,loss4], axis=1)
                Spi_dataset = Spi_dataset.append(s, ignore_index=True)

            ## AMO call
            # print('----- AMO 2 ----------')


            Dpimin_dataset1 = Dpimin_dataset.iloc[0:T1,:]
            Dpimin_dataset2 = Dpimin_dataset.iloc[T1:,:].drop(columns=['label'])
            # print('Dpimin_dataset1', Dpimin_dataset1)
            # print('Dpimin_dataset2', Dpimin_dataset2)

            pi = Argmin.argmin(self.eps, self.nu, self.fairness, Dpimin_dataset1, Dpimin_dataset2)

            self.num_amo_calls += 1
            # print('----- END AMO 2 ----------')

            ## This is mostly to make sure we have the predictions cached for this new policy
            if pi not in q_losses.keys():
                (tmp, predictions) = self.get_cum_loss(H, pi, predictions)
                q_losses[pi] = tmp
                # if q_losses[pi] < leader_loss:
                #     best_pi = pi
                #     leader_loss = q_losses[pi]

            assert pi in predictions.keys(), "Uncached predictions for new policy pi"

            (Dpi, predictions) = self.get_cum_loss(Dpinormal_dataset, pi, predictions)
            Dpi = Dpi - 4 + (leader_loss / (psi * t * mu))

            if Dpi > 0:
                updated = True

                Vpi, ptwo = self.get_cum_loss(Vpi_dataset, pi, predictions)
                Spi, ptwo = self.get_cum_loss(Spi_dataset, pi, predictions)

                toadd = (Vpi + Dpi) / (2 * (1 - mu) * Spi)
                Q.append((pi, toadd))
                print('--- EVALUATION -- policy high variance, update loop:', iterations)
                self.statistics_var.evaluate(pi.model)
            else:
                print('----- END SOLVE OP naturally ----------')
                return Q

        print('----- END SOLVE OP ----------')
        return Q

    def get_cum_loss(self, dataset, pi, predictions):
        # H: DF, best_pi: Policy, pred: dict, t:int, feat:DF

        # print('predictions', dataset.index)
        # print('pi.get_all_decisions(dataset.iloc[:, 0:2])', pi.get_all_decisions(dataset.iloc[:, 0:2]))

        if pi not in predictions.keys():
            ## This is going to go horribly wrong if dataset is not the right size
            # assert len(dataset) == t, "If predictions not yet cached, dataset should have len = self.t"
            # predictions[pi] = dict(zip([i for i in dataset.index], pi.get_all_decisions([row for index, row in dataset.iloc[:,0:2].iterrows()])))
            # Todo FIX THIS: danger with other datasets selecting only first two columns
            predictions[pi] = dict(zip([i for i in dataset.index], pi.get_all_decisions(dataset.loc[:, ['features', 'sensitive_features_X']])))

        score = 0.0
        for x in dataset.index:
            l = dataset.filter(items=['l0', 'l1']).loc[x,:]
            l = l.rename({'l0': 0, 'l1': 1})
            pred, prob = predictions[pi][x]
            score += l.loc[pred]*prob

        return score, predictions


