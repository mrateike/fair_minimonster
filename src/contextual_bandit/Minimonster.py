import time
from pathlib import Path
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.contextual_bandit import Argmin
from src.evaluation.Evaluation import Evaluation, save_and_plot_results, my_plot
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
        self.statistics_loss = Evaluation(TT, test_seed)
        self.statistics_var = Evaluation(TT, test_seed)
        self.statistics_final = Evaluation(TT, test_seed)
        self.loss = []
        self.opt_loss = []
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

        self.path = path




    def fit(self, T2, T1, batch, batchsize):

        XA = pd.DataFrame()
        A = pd.Series(name='sensitive_features')
        L = pd.DataFrame(columns=['l0', 'l1'])
        dataset2_collected = pd.DataFrame()

        l1 = []
        l2 = []

        # for _solve_Opt
        psi = 4*(math.e -2)*np.log(T2+T1)


        # predictions = {}
        Q = []
        # best_pi = None

        loss_path = "{}/loss_results".format(self.path)
        Path(loss_path).mkdir(parents=True, exist_ok=True)

        self.var_path = "{}/var_results".format(self.path)
        Path(self.var_path).mkdir(parents=True, exist_ok=True)


        print('--- EVALUATION --  first best policy on phase 1 data')
        # print('self.dataset1', self.dataset1)
        best_pi, results_dict_loss  = Argmin.argmin(loss_path, self.statistics_loss, self.eps, self.nu, self.fairness, self.dataset1)

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
        elif batch == 'warm':
           training_points = [3,5]
           m +=2
           while True:
               training_points.append(int(m ** 2))
               m += 1
               if int(m ** 2) > T2:
                   break

        elif batch == 'none':
            # need to collect a few first such that we do not get a label problem
            m+=20
            while True:
                training_points.append(int(m))
                m += 1
                if int(m) > T2:
                    break
        else:
            print('ERROR in batches')




        #----------- run loop --------------
        m=0
        t = 1
        while t < T2 + 1:
            m+=1

            individual = self.dataset2.iloc[(t-1)]

            xa = individual.loc[['features', 'sensitive_features_X']].to_frame().T
            a = pd.Series(individual.loc['sensitive_features'], name='sensitive_features', index=[individual.name])

            XA = XA.append(xa)
            A = A.append(a)

            y = individual.loc['label']
            d, p = self.sample(xa, Q, best_pi, m)
            l = self.B.get_loss(d, y)



            # IPS loss, attention: order of columns might change
            # sometimes [l1, l0], sometimes [l0, l1]
            ips_loss = pd.DataFrame()
            ips_loss.at[individual.name,str(d)] = l / p
            ips_loss.at[individual.name, str(1-d)] = 0
            ips_loss = ips_loss.rename(columns={'0':'l0','1':'l1'})


            L = L.append(ips_loss)




            # ---------- update batch ( push new ones) --------
            if t in training_points:

                # data collected in this batch
                dataset_batch = pd.concat([XA, L, A], axis=1)

                # data collected so far in phase 2
                dataset2_collected = dataset2_collected.append(dataset_batch)


                print('--- EVALUATION -- batch update ', m, 'at time', t, 'best policy')
                # print('dataset2_collected',dataset2_collected)
                best_pi, results_dict_loss = Argmin.argmin(loss_path, self.statistics_loss, self.eps, self.nu, self.fairness, self.dataset1, dataset2_collected)

                self.num_amo_calls += 1


                # data collected from phase 1 and phase 2 so far
                self.history = self.history.append(dataset_batch, ignore_index=True)

                # update Q
                Q = self._solve_op(self.history, T1, m, best_pi, psi)

                # clear batch containers
                XA = pd.DataFrame()
                A = pd.Series(name='sensitive_features')
                L = pd.DataFrame()
                # ---------- END  batch update --------

            # ----- evaluate loss -----------
            self.loss.append(l)
            l1.append(np.sum(self.loss) / t)

            # Todo: implement loss and regret computation
            # in hindsight, best policy loss
            # xa is DF
            # if best_pi != None:
            #     print('xs', xa)
            #     pred = best_pi.get_decision(xa)
            #     print('pred', pred)
            #     print('full_lvec', full_lvec)
            #     leader_loss = full_lvec.loc[pred]
            #     print('leader_loss', leader_loss)
            #     print('type(self.opt_loss) ', type(self.opt_loss))
            #     self.opt_loss = self.opt_loss.append(leader_loss)
            #     print('self.opt_loss ', self.opt_loss)
            #     l2 = l2.append(self.opt_loss/ t)
            # else:
            #     # check again
            #     l2 = l2

            # ----- next loop
            t += 1

        # ----- End fit: print results------------
        l2 = 0




        # ------ evaluation of the finally best policy returned -------
        final_path = "{}/final_results".format(self.path)
        Path(final_path).mkdir(parents=True, exist_ok=True)

        self.statistics_final.evaluate(best_pi)
        decisions = self.statistics_final.scores_array
        a_test = self.statistics_final.a_test.to_frame().to_numpy()
        y_test = self.statistics_final.y_test.to_frame().to_numpy()
        updates = len(self.statistics_final.acc_list_overall)

        print('---- Floyds stats ----')
        # ------ statistics from Floyd -------
        floyds_stats = Statistics(
            predictions=decisions,
            protected_attributes=a_test,
            ground_truths=y_test,
            additonal_measures= {UTILITY: {'measure_function': lambda s, y, decisions : np.mean(decisions * (y - 0.5)),
            'detailed': False}})

        save_and_plot_results(
            base_save_path=final_path,
            statistics=floyds_stats, update_iterations = updates)

        print('---- My stats ----')
        # my stats
        self.statistics_final.save_plot_process_results(results_dict_loss, final_path)


        # ------ retun statement ------
        # return l1, l2, Q, best_pi














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

    def sample(self, x, Q, best_pi, t):
        # xa = pd.DataFrame
        # Q = [Policy,float]
        # best_pi = Policy
        # print('x', x, type(x))

        p = np.zeros(2)
        if Q == []:
            dec_prob = best_pi.predict(x)
            dec = int(dec_prob[0,0])
            prob = dec_prob[0,1]
            p[dec] = prob
            # print('p', p)

        else:
            # print('Q', Q)
            for item in Q:
                # print('item', item)
                pi = item[0]
                w = item[1]
                # print('w in Q', w)
                dec_prob = pi.get_decision(x).squeeze()
                # print('dec_prob', dec_prob)
                dec = int(dec_prob[0])
                # print('dec', dec)
                prob = dec_prob[1]
                p[dec] += w*prob
                p[1-dec]+=w*(1-prob)
                # print('p', p)

            ## Mix in leader

            # print('best_pi', best_pi)
            bp_dec_prob = best_pi.get_decision(x).squeeze()
            # print('best_pi_get_dec', bp_dec_prob)
            bp_dec = int(bp_dec_prob[0])
            bp_prob = bp_dec_prob[1]
            # print('bp_prob', bp_prob)
            # print('bp_dec', bp_dec)

            # print('Q', Q)
            bp_w = 1 - np.sum([q[1] for q in Q])
            # print('bp_w ', bp_w)
            # print('bp_prob', bp_prob)
            # print('p', p)
            p[bp_dec] += (bp_w*bp_prob)
            p[1-bp_dec] +=bp_w*(1-bp_prob)
            # print('add to one p', p)

        # in paper:
        # print('p before mixing in', p)
        # print('mu(t)', self._get_mu(t))
        p = (1 - 2 * self._get_mu(t)) * p + self._get_mu(t)
        # print('p after mixing in', p)
        ## Take decision

        # assert p[1] >= 0 and p[1] <= 1 and p[1]+p[0] <=1, 'probability needs to be [0,1]'


        dec = (p[1] >= np.random.rand(1)) * 1
        dec = dec.squeeze()

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
        while updated and iterations < 5:
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
            print('--- EVALUATION -- policy high variance, update loop:', iterations)

            Dpimin_dataset1 = Dpimin_dataset.iloc[0:T1,:]
            Dpimin_dataset2 = Dpimin_dataset.iloc[T1:,:].drop(columns=['label'])
            # print('Dpimin_dataset1', Dpimin_dataset1)
            # print('Dpimin_dataset2', Dpimin_dataset2)

            pi, results_dict_var = Argmin.argmin(self.var_path, self.statistics_var, self.eps, self.nu, self.fairness, Dpimin_dataset1, Dpimin_dataset2)
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
                # print('---- Second Constraint: broken -----')
                # print('Vpi_dataset', Vpi_dataset)
                # print('Vpi_dataset.columns', Vpi_dataset.columns)
                # print('Vpi_dataset.index', Vpi_dataset.index)
                Vpi, ptwo = self.get_cum_loss(Vpi_dataset, pi, predictions)

                # print('Spi_dataset', Spi_dataset)
                # print('Spi_dataset.columns', Spi_dataset.columns)
                # print('Spi_dataset.index', Spi_dataset.index)
                Spi, ptwo = self.get_cum_loss(Spi_dataset, pi, predictions)

                toadd = (Vpi + Dpi) / (2 * (1 - mu) * Spi)
                Q.append((pi, toadd))
            # else:
            #     print('----- END SOLVE OP naturally ----------')
            #     return Q, best_pi

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


        # print('predictions', predictions)
        score = 0.0
        # dataset.index.names = [phase, i]
        for x in dataset.index:
            l = dataset.filter(items=['l0', 'l1']).loc[x,:]
            l = l.rename({'l0': 0, 'l1': 1})
            pred, prob = predictions[pi][x]
            score += l.loc[pred]*prob

        return score, predictions


