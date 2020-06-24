import time
from pathlib import Path
import matplotlib as mpl
# mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.contextual_bandit import Argmin
from src.evaluation.Evaluation import Evaluation, my_plot, get_average_regret
from src.evaluation.training_evaluation import Statistics
from data.util import save_dictionary
from src.evaluation.training_evaluation import UTILITY
import math
import time


import os

my_path = os.path.abspath(__file__)  # Figures out the absolute path for you in case your working directory moves around.


class MiniMonster(object):
    """
    Implementation of MiniMonster with a scikit_learn learning algorithm as the AMO.
    """

    def __init__(self, B, fairness, eps, nu, TT, seed, path, mu, num_iterations):
        self.B = B
        # XA, L, A, Y
        # self.dataset1 = dataset1
        # self.dataset2 = dataset2.drop(['l0', 'l1'], axis=1)
        self.num_iterations = num_iterations



        self.seed = seed
        # loss_path = "{}/loss_policies_results".format(path)
        # Path(loss_path).mkdir(parents=True, exist_ok=True)
        # loss_path = "{}/loss_".format(loss_path)

        # var_path = "{}/var_policies_results".format(path)
        # Path(var_path).mkdir(parents=True, exist_ok=True)
        # var_path = "{}/var_".format(var_path)

        # dec_path = "{}/bandit_decions_results".format(path)
        # Path(dec_path).mkdir(parents=True, exist_ok=True)


        # self.process_path = "{}/bandit_regret_results".format(path)
        # Path(self.process_path).mkdir(parents=True, exist_ok=True)

        # self.regret_path = "{}/bandit_regret_results".format(path)
        # Path(self.regret_path).mkdir(parents=True, exist_ok=True)
        #
        # self.statistics_loss = Evaluation(TT, test_seed, loss_path, B, x_label)
        # self.statistics_var = Evaluation(TT, test_seed, var_path, B, x_label)

        x_label = 'individuals'

        dec_path = "{}/bandit_".format(path)
        test_seed = 45*seed

        self.statistics_decisions = Evaluation(TT, test_seed, dec_path, B, x_label)

        self.varOracle_time_list = []
        self.num_varOracle_calls = 0
        self.num_lossOracle_calls = 0
        self.path = path

        self.mu = mu

        # XA, L, A (supervised)

        # print('history', self.history)
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
        self.lossOracle_time_list = []
        self.list_num_policies_added = []
        self.list_cum_loss_total_time = []
        self.best_loss_t = []
        self.loop_time_list = []
        self.num_update_Q = 0




    # def fit(self, T2, T1, batch, batchsize):
    def fit(self, dataset, alpha, batch, batchsize):
        self.real_loss = []
        # best_loss_t = []


        # loss_history = pd.DataFrame()

        T = dataset.shape[0]
        T1 = int(round(T ** (2 * alpha),0))
        T2 = T-T1
        print('T', T)
        print('T1', T1)
        print('T2', T2)
        self.T = T
        rand = np.random.RandomState(21 * self.seed)
        self.randomthresholds = rand.rand(T)

        rand1 = np.random.RandomState(27 * self.seed)
        self.randomthresholds1 = rand1.rand(T)
        rand2 = np.random.RandomState(29 * self.seed)
        self.randomthresholds2 = rand2.rand(len(self.statistics_decisions.XA_test))


        # print('RAND THRESHOLDS', self.randomthresholds)

        # ------------- batch settings
        training_points = []
        m = 1
        if batch == "exp":
            while True:
                training_points.append(int(2 ** (m - 1)))
                m += 1
                if (2 ** (m - 1)) > T2:
                    break
        elif batch == 'lin':
            while True:
                batchsize = int(batchsize)
                training_points.append(int(m * batchsize))
                m += 1
                if int(m * batchsize) > T2:
                    break
        elif batch == 'warm_start':
            training_points = [3, 5]
            m += 2
            while True:
                training_points.append(int(m ** 2))
                m += 1
                if int(m ** 2) > T2:
                    break

        elif batch == 'none':
            # need to collect a few first such that we do not get a label problem
            m = 1
            while True:
                training_points.append(int(m))
                m += 1
                if int(m) > T2:
                    break
        else:
            print('ERROR in batches')

        # print('training_points', training_points)

        # ----------- run loop --------------

        history = dataset.iloc[:T1]
        # print('history', history)


        #----IPS & non IPS Regret
        self.real_loss.extend(history.loc[:,'l1'].tolist())


        scores=pd.Series(np.ones(len(self.statistics_decisions.y_test)))
        self.statistics_decisions.evaluate_scores(scores)


        # for _solve_Opt
        psi = 4*(math.e -2)*np.log(T)

        m = 0
        t = 0
        self.x_axis = [T1]
        while t <= T:

            if m == 0:
                batchsize = 0
            # print('update: m', m, 't', t, 'batchsize', batchsize)
            Q, best_pi = self.update(history, T1, psi, batchsize, m)

            if t == T2:
                break



            # print('time t ', t)
            # print('batch m ', m)
            # print('training_points ', training_points)
            # print('T2', T2)

            if m < len(training_points):
                batchpoint = training_points[m]
                batchsize = batchpoint - t
            else:
                batchsize = T2-t
                batchpoint = T2

            individual = dataset.iloc[(T1+t):(T1+batchpoint)]
            # print('individual', individual)

            t = batchpoint


            xa = individual.loc[:,['features', 'sensitive_features']]

            # print('sample: m', m, 't', t, 'xa', xa)
            d, p = self.sample(xa, Q, best_pi, m)
            p = pd.DataFrame(p, index=individual.index)
            y = individual.loc[:,'label']
            l = self.B.get_loss(d, y)

            lip = l.div(p).to_numpy()
            lip[list(range(0, len(lip))),1-d] = 0

            ips_loss = pd.DataFrame(lip, index=l.index)

            # --- IPS Regret ----
            # self.real_loss.extend(ips_loss.lookup(ips_loss.index, d).tolist())
            # --- non IPS Regret
            self.real_loss.extend(l.lookup(l.index, d).tolist())

            ips_loss = ips_loss.rename(columns={0:'l0', 1:'l1'})
            dataset_update = pd.concat([individual.loc[:,['features', 'sensitive_features', 'label']],ips_loss], axis =1)
            history = pd.concat([history, dataset_update], axis = 0, ignore_index=True)


            if m == len(training_points):
                self.x_axis.append(T)
            else:
                self.x_axis.append((training_points[m] + T1))

            m += 1

        # ------- end of fitting -----------

        print('x-axis', self.x_axis)
        print('ACC', self.statistics_decisions.ACC_list)
        print('DP', self.statistics_decisions.DP_list)
        print('TPR', self.statistics_decisions.TPR_list)
        data_decisions = {'ACC': self.statistics_decisions.ACC_list, \
                     'DP': self.statistics_decisions.DP_list, \
                     'TPR': self.statistics_decisions.TPR_list,
                      'X' : self.x_axis}
        parameter_save_path = "{}/evaluation.json".format(self.path)
        save_dictionary(data_decisions, parameter_save_path)


        # ----- Calculate roundwise regret ------------
        timesteps = range(1, len(self.real_loss) + 1)

        cum_real_loss = np.cumsum(self.real_loss)
        exp_real_loss = cum_real_loss / timesteps
        cum_exp_real_loss = np.cumsum(exp_real_loss)
        cum_best_loss_t = np.cumsum(self.best_loss_t)
        exp_best_loss = cum_best_loss_t / timesteps
        cum_exp_best_loss = np.cumsum(exp_best_loss)

        reg = cum_exp_real_loss - cum_exp_best_loss
        reg[reg < 0] = 0

        # ----- Calculate regret T in hindsight ------------
        xa = history.loc[:, ['features', 'sensitive_features']]
        ips = history.loc[:, ['l0', 'l1']]
        db, _ = best_pi.get_decision(xa)
        ips.columns = range(ips.shape[1])
        best_loss_T = ips.lookup(ips.index, db).tolist()

        cum_best_loss_T = np.cumsum(best_loss_T)
        exp_best_loss_T = cum_best_loss_T / timesteps
        cum_exp_best_loss_T = np.cumsum(exp_best_loss_T)

        regT = cum_exp_real_loss - cum_exp_best_loss_T
        regT[regT < 0] = 0

        reg0_dict = {}
        reg0_dict['regt_cum'] = reg.tolist()
        reg0_dict['regT_cum'] = regT.tolist()
        # reg0_dict['x_axis'] = x_axis

        regret_path = "{}/regret.json".format(self.path)
        save_dictionary(reg0_dict, regret_path)

        x_axis = [0]
        x_axis.extend(training_points)
        # print('training_points')
        x_axis = np.array(x_axis) + T1
        x_axis = x_axis.tolist()
        if training_points[-1] < T2:
            x_axis.extend([T])



        # --- end of fit -------
        # cum_loop_time = np.cumsum(self.loop_time_list).tolist()
        # cum_varOracle_time = np.cumsum(self.varOracle_time_list).tolist()

        # process_data = {}
        # process_data['cum-loop-time'] = cum_loop_time
        # process_data['cum-varOracle-time'] = cum_varOracle_time
        # print('loop_time_list', loop_time_list)
        # process_data['list-loop-time'] = self.loop_time_list
        # process_data['cum-lossOracle-time'] = cum_lossOracle_time.tolist()
        # print('self.lossOracle_time_list', self.lossOracle_time_list)
        # process_data['list-lossOracle-time'] = self.lossOracle_time_list
        # print('self.varOracle_time_list', self.varOracle_time_list)
        # process_data['list-varOracle-time'] = self.varOracle_time_list
        # process_data['list_cum_loss_total_time'] = self.list_cum_loss_total_time

        # print('num_lossOracle_calls', self.num_lossOracle_calls)
        # print('self.num_varOracle_calls', self.num_varOracle_calls)
        # print('self.list_num_policies_added', self.list_num_policies_added)
        # print('training points', training_points)

        # process_data['num_lossOracle_calls'] = self.num_lossOracle_calls
        # process_data['num_varOracle_calls'] = self.num_varOracle_calls
        # process_data['num_var_policies added'] = self.list_num_policies_added
        #
        # process_path = "{}/process_measures.json".format(self.process_path)
        # save_dictionary(process_data, process_path)




        # print('x_axis length', len(x_axis ))
        # print('x_axis', x_axis)
        # print('self.num_update_Q', self.num_update_Q )

        # plot_dict = {}
        # plot_dict['x_axis_time'] = x_axis
        # plot_dict['x_axis_reg'] = timesteps
        # plot_dict['x_label'] = 'individuals'
        # plot_dict['y_label0'] = 'update-Q time / update'
        # plot_dict['y_label1'] = 'process IPS Reg_t0'
        # plot_dict['y_label2'] = 'in loop: cum varOracle_time / update'
        # plot_dict['y_label3'] = 'hindsight IPS Reg_T0'
        # plot_dict['square'] = 'NO'
        # plot_dict['process'] = 'YES'
        # plot_dict['evaluation'] = 'NO'
        # process_path = "{}/process_".format(self.process_path)
        #
        # my_plot(process_path, plot_dict, cum_loop_time, \
        #         reg, cum_varOracle_time, regT)
        #


    def update(self, history, T1, psi, batch_size, m):

        dataset1 = history.loc[:T1-1]
        dataset2 = history.loc[T1:].drop(['label'], axis=1)

        best_pi = Argmin.argmin(self.randomthresholds, self.eps, self.nu, self.fairness, dataset1, dataset2)

        batch = history.iloc[-batch_size:]

        batch_xa = batch.loc[:,['features', 'sensitive_features']]
        batch_best_decisions, _ = best_pi.predict(batch_xa)
        batch_best_decisions = pd.Series(batch_best_decisions).astype(int)




        # --- IPS regret
        # df = batch.loc[:, ['l0', 'l1']]
        # df.columns = range(df.shape[1])
        # self.best_loss_t.extend(df.lookup(df.index, batch_best_decisions).tolist())
        # --- non IPS regret
        y = batch.loc[:, 'label']

        l = self.B.get_loss(batch_best_decisions, y)
        self.best_loss_t.extend(l.lookup(l.index, batch_best_decisions).tolist())


        # print('--- EVALUATION -- batch update ', m, 'best_policy')
        # self.statistics_loss.evaluate(best_pi.model, self.x_axis)

        # start = time.time()
        Q = self._solve_op(history, T1, m, best_pi, psi)
        # stop = time.time()
        # loop_time = np.array([stop - start])
        # self.num_update_Q +=1
        # self.loop_time_list.append(loop_time[0])


        return Q, best_pi

    def sample(self, xa, Q, best_pi, m):
        # xa = pd.DataFrame
        # Q = [Policy,float]
        # best_pi = Policy

        pdec = np.zeros((xa.shape[0], 2))

        xa_test = self.statistics_decisions.XA_test

        pdt = np.zeros((xa_test.shape[0], 2))


        for item in Q:
            pi = item[0]
            w = item[1]

            _, p = pi.get_decision(xa)
            pdec = np.add(pdec, w * p)

            _, pt = pi.predict(xa_test)

            pdt = np.add(pdt, w * pt)

        ## Mix in leader
        w_bp = 1 - np.sum([q[1] for q in Q])
        mu = self._get_mu(m)

        _, pb = best_pi.get_decision(xa)
        pdec = np.add(pdec, w_bp * pb)
        pdec = (1 - 2 * mu) * pdec + mu

        _, pbt = best_pi.get_decision(xa_test)
        pdt = np.add(pdt, w_bp * pbt)
        pdt = (1 - 2 *  mu) * pdt + mu

        threshold1 = self.randomthresholds1[xa.index[0]:xa.index[-1] + 1]
        dec = (pdec[:,1] >= threshold1) * 1
        dec = dec

        threshold2 = self.randomthresholds2[xa_test.index[0]:xa_test.index[-1] + 1]
        dect = (pdt[:, 1] >= threshold2) * 1
        dect = dect.squeeze()

        scores_test = pd.Series(dect)
        # print('--- EVALUATION -- learners decision making')
        # print('scores_test', scores_test)
        # print('self.x_axis', self.x_axis)
        self.statistics_decisions.evaluate_scores(scores_test)

        return dec, pdec

    def _get_mu(self, t):
        """
        Return the current value of mu_t
        """
        # a = 1.0/(4)
        # b = np.sqrt(np.log(16.0*(self.t**2)*self.B.N/self.delta)/float(4*self.t))
        a = self.mu
        # print('t', t)
        if t == 0:
            b = np.inf
        else:
            b = self.mu * np.sqrt(2) / np.sqrt(t)
        c = np.min([a, b])
        return np.min([1, c])
        # return 0.1

    def _solve_op(self, H, T1, m, best_pi, psi, Q=None):

        """
        Main optimization logic for MiniMonster.
        """
        # print('----- BEGIN SOLVE OP ----------')
        num_policies_added = 0
        # pi_var_total_time = 0

        mu = self._get_mu(m)

        t = H.shape[0]

        # Todo: implement warm start option
        Q = []  ## self.weights

        predictions = {}
        q_losses = {}

        leader_loss, predictions = self.get_cum_loss(H, best_pi, predictions)

        iterations = 0


        while iterations < int(self.num_iterations):
            iterations += 1

            score = np.sum([item[1] * (4 + ((q_losses[item[0]] - leader_loss) / (psi * t * mu))) for item in Q])

            if score > 4:
                c = 4 / score
                Q = [(item[0], c * item[1]) for item in Q]

            q = np.zeros((len(H),2))
            for item in Q:
                pi = item[0]
                w = item[1]
                prob_dec = predictions[pi].iloc[:,1].to_numpy()
                dec = predictions[pi].iloc[:,0].to_numpy()
                p1 = (dec)*(prob_dec) + (1-dec)*(1-prob_dec)
                p0 = 1-p1
                p = np.stack((p0, p1), axis=1)
                q = np.add(q, w * p)

            q = (1.0 - 2 * mu) * q + (mu)

            v = 1.0 / (t * q)
            s = 1.0 / (t * (q ** 2))

            loss = H.loc[:,['l0', 'l1']].to_numpy()

            # -bt:
            reg = loss - leader_loss
            reg[reg < 0] = 0
            bt = reg / (t * psi * mu)

            # -l:
            # bt = loss

            _H = H.loc[:, ['features', 'sensitive_features', 'label']]

            loss1 = pd.DataFrame(bt - v, columns=['l0', 'l1'], index=H.index)
            Dpimin_dataset = pd.concat([_H, loss1], axis=1)

            loss2 = pd.DataFrame(v - bt, columns=['l0', 'l1'], index=H.index)
            Dpinormal_dataset = pd.concat([_H, loss2], axis=1)

            Dpimin_dataset1 = Dpimin_dataset.iloc[0:T1,:]
            Dpimin_dataset2 = Dpimin_dataset.iloc[T1:,:].drop(columns=['label'])
            # print('Dpimin_dataset1', Dpimin_dataset1)
            # print('Dpimin_dataset2', Dpimin_dataset2)

            # start = time.time()
            pi = Argmin.argmin(self.randomthresholds, self.eps, self.nu, self.fairness, Dpimin_dataset1, Dpimin_dataset2)
            # stop = time.time()
            # pi_var_total_time = pi_var_total_time + np.array([stop - start])[0]
            # self.num_varOracle_calls += 1

            if pi not in q_losses.keys():
                pi_loss, _ = self.get_cum_loss(H, pi, predictions)
                q_losses[pi] = pi_loss

            assert pi in predictions.keys(), "Uncached predictions for new policy pi"


            Dpi, _ = self.get_cum_loss(Dpinormal_dataset, pi, predictions)

            # -bt:
            Dpi = Dpi - 4
            # l: Dpi = Dpi - ( 4 + leader_loss)

            if Dpi > 0:
                loss3 = pd.DataFrame(v, columns=['l0', 'l1'], index=H.index)
                Vpi_dataset = pd.concat([_H, loss3], axis=1)

                loss4 = pd.DataFrame(s, columns=['l0', 'l1'], index=H.index)
                Spi_dataset = pd.concat([_H, loss4], axis=1)

                Vpi, _ = self.get_cum_loss(Vpi_dataset, pi, predictions)
                Spi, _ = self.get_cum_loss(Spi_dataset, pi, predictions)

                toadd = (Vpi + Dpi) / (2 * (1 - 2*mu) * Spi)
                Q.append((pi, toadd))
                num_policies_added +=1

            else:
                # print('----- END SOLVE OP naturally ----------')
                # self.varOracle_time_list.append(pi_var_total_time)
                self.list_num_policies_added.append(num_policies_added)
                return Q

        # print('----- END SOLVE OP ----------')
        # self.varOracle_time_list.append(pi_var_total_time)
        self.list_num_policies_added.append(num_policies_added)
        return Q

    def get_cum_loss(self, dataset, pi, predictions):
        # H: DF, best_pi: Policy, pred: dict, t:int, feat:DF

        if pi not in predictions.keys():
            # Todo FIX THIS: danger with other datasets selecting only first two columns
            values = pi.get_all_decisions(dataset.loc[:, ['features', 'sensitive_features']])
            predictions[pi] = values

        prob = predictions[pi][1]
        dec = predictions[pi][0]
        loss_df = dataset.loc[:,['l0', 'l1']]

        loss_df.columns = range(loss_df.shape[1])
        loss = pd.Series(loss_df.lookup(loss_df.index, dec))
        score = loss.dot(prob)
        return score, predictions


