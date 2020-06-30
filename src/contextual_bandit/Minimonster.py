# Copyright (c) 2016 akshaykr, adapted by mrateike
import numpy as np
import pandas as pd
from src.contextual_bandit import Argmin
from src.evaluation.Evaluation import Evaluation
from data.util import save_dictionary
import math

import os

my_path = os.path.abspath(__file__)  # absolute path

"""
Implementation adapted from akshaykr (https://github.com/akshaykr/oracle_cb)
This is the main algorithm Minimonster (coordinate descent) from 
Agarwal, et al. (2014).  Taming the monster: A fast and simple algorithm for contextual bandits. 
In International Conference on Machine Learning (pp. 1638-1646).
Adapted for the fair contextual bandit approach by 

"""



class FairMiniMonster(object):
    """A contextualbandit algorithm which implements the fair contextual bandit algorithm,
        described in detail by
        Bechavod, Y., Ligett, K., Roth, A., Waggoner, B., & Wu, S. Z. (2019).
        Equal opportunity in online classification with partial feedback.
        In Advances in Neural Information Processing Systems (pp. 8974-8984).


        """

    def __init__(self, B, fairness, eps, nu, TT, seed, path, mu, num_iterations):
        """
        Initialization of Minimonster algorithm

         Parameters
        ----------
        B: Simulator
            simulator class to generate training and test datsets
        fairness : str
            type of fairness (DP, EO)
        eps: float
            fairness relaxation parameter, value > 0
        nu: float
            accuracy parameter, value > 0
        TT: int
            total number of test data to be generated
        seed: int
            to fix seed
        path: str
            path directory to save results
        mu: float
            minimum probability for smoothed distribution
        num_iterations: int
            maximum number of iterations of coordinate descent loop

        """

        self.B = B
        self.num_iterations = num_iterations
        self.seed = seed
        self.mu = 0.1
        self.fairness = fairness
        self.eps = eps
        self.nu = nu
        self.path = path
        self.mu = mu

        x_label = 'individuals'
        dec_path = "{}/bandit_".format(path)

        # get random test seed
        test_seed = 45*seed

        # initialize a statistics object for evaluation of pi
        self.statistics_decisions = Evaluation(TT, test_seed, dec_path, B, x_label)

        self.best_loss_t = []
        self.list_num_policies_added = []




    def fit(self, dataset, alpha, batch, batchsize):
        """ function to learn distribution Q over T rounds
        evaluates the distribution and learning and saves results
        in .json files, has no return

        Parameters
        ----------
        dataset : pd.DataFrame
            total dataset
            size: T datapoints - phase 1 and phase 2 dataset (combined)
        alpha: float
            dataset splitting parameter, i.e. T1 = T^{2alpha}
            values: [0.25, 0.5]
        batch: str
            batch type ('none', 'lin', 'exp')
        batchsize : int
            batch size for lin batch type (default 1 if lin or exp)

        """

        self.real_loss = []



        # calculation of phase 1 / phase 2 split
        # given alpha and T
        T = dataset.shape[0]
        T1 = int(round(T ** (2 * alpha),0))
        T2 = T-T1
        print('T', T)
        print('T1', T1)
        print('T2', T2)
        self.T = T

        # fix seeds
        rand = np.random.RandomState(21 * self.seed)
        self.randomthresholds = rand.rand(T)
        rand1 = np.random.RandomState(27 * self.seed)
        self.randomthresholds1 = rand1.rand(T)
        rand2 = np.random.RandomState(29 * self.seed)
        self.randomthresholds2 = rand2.rand(len(self.statistics_decisions.XA_test))


        # set batch settings
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
            m = 1
            while True:
                training_points.append(int(m))
                m += 1
                if int(m) > T2:
                    break
        else:
            print('ERROR in batches')


        # ----------- run loop --------------

        # phase 1 data added to history
        history = dataset.iloc[:T1]


        # save loss for regret calculation later
        self.real_loss.extend(history.loc[:,'l1'].tolist())

        # evaluate phase 1 decisions (d=1)
        scores=pd.Series(np.ones(len(self.statistics_decisions.y_test)))
        self.statistics_decisions.evaluate_scores(scores)

        # set values for _solve_Opt
        psi = 4*(math.e -2)*np.log(T)

        m = 0
        t = 0
        self.x_axis = [T1]

        # over all T
        while t <= T:

            if m == 0:
                batchsize = 0

            # UPDATE Q and best_pi
            Q, best_pi = self.update(history, T1, psi, batchsize, m)

            if t == T2:
                break

            if m < len(training_points):
                batchpoint = training_points[m]
                batchsize = batchpoint - t
            else:
                batchsize = T2-t
                batchpoint = T2

            # environment draws new context-ground truth pair
            individual = dataset.iloc[(T1+t):(T1+batchpoint)]

            t = batchpoint

            # context of new sample
            xa = individual.loc[:,['features', 'sensitive_features']]

            # SAMPLE: learner decision given context, Q and best_pi
            # returns decision and probability with which decision was taken
            d, p = self.sample(xa, Q, best_pi, m)
            p = pd.DataFrame(p, index=individual.index)

            # ground truth label of sample
            y = individual.loc[:,'label']

            # get transformed loss vector of decision and grund truth label
            l = self.B.get_loss(d, y)

            # get IPS loss
            lip = l.div(p).to_numpy()
            lip[list(range(0, len(lip))),1-d] = 0
            ips_loss = pd.DataFrame(lip, index=l.index)

            self.real_loss.extend(l.lookup(l.index, d).tolist())

            ips_loss = ips_loss.rename(columns={0:'l0', 1:'l1'})
            dataset_update = pd.concat([individual.loc[:,['features', 'sensitive_features', 'label']],ips_loss], axis =1)

            # update history
            history = pd.concat([history, dataset_update], axis = 0, ignore_index=True)


            if m == len(training_points):
                self.x_axis.append(T)
            else:
                self.x_axis.append((training_points[m] + T1))

            m += 1

        # ------- end of fitting -----------
        #  saving data
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

        reg0_dict = {}
        reg0_dict['regt_cum'] = reg.tolist()

        regret_path = "{}/regret.json".format(self.path)
        save_dictionary(reg0_dict, regret_path)

        x_axis = [0]
        x_axis.extend(training_points)
        x_axis = np.array(x_axis) + T1
        x_axis = x_axis.tolist()
        if training_points[-1] < T2:
            x_axis.extend([T])

        # --- end of fit -------



    def update(self, history, T1, psi, batch_size, m):
        """ function to learn new Q and best_pi at updating round m
        returns: distribution Q, policy best_pi

        Parameters
        ----------
        history : pd.DataFrame
            history H_t with all samples seen so until time t (phase 1 and phase2)
        T1: int
            size of phase 1
        psi: float
            regularization parameter for regret b_t(pi)
        batch_size : int
            amount of new samples using in this updating step
                constant, if batchtype = 'lin'
                exponentially growing, if batchtype = 'exp'
                1, if batchtype = 'none'
        m : int
            updating step, if batchtype = 'none' then m = t

        """

        dataset1 = history.loc[:T1-1]
        dataset2 = history.loc[T1:].drop(['label'], axis=1)

        # call fair orcale to obtain best_pi minimizing loss
        best_pi = Argmin.argmin(self.randomthresholds, self.eps, self.nu, self.fairness, dataset1, dataset2)

        batch = history.iloc[-batch_size:]

        batch_xa = batch.loc[:,['features', 'sensitive_features']]

        # evaluate returned best_pi
        batch_best_decisions, _ = best_pi.predict(batch_xa)
        batch_best_decisions = pd.Series(batch_best_decisions).astype(int)


        y = batch.loc[:, 'label']

        l = self.B.get_loss(batch_best_decisions, y)
        self.best_loss_t.extend(l.lookup(l.index, batch_best_decisions).tolist())

        # solve coordinate descent problem
        Q = self._solve_op(history, T1, m, best_pi, psi)


        return Q, best_pi

    def sample(self, xa, Q, best_pi, m):
        # xa = pd.DataFrame
        # Q = [Policy,float]
        # best_pi = Policy

        """ function to learn new Q and best_pi at updating round m
           returns: Q, best_pi

           Parameters
           ----------
           history : pd.DataFrame
               history H_t with all samples seen so until time t (phase 1 and phase2)
           T1: int
               size of phase 1
           psi: float
               regularization parameter for regret b_t(pi)
           batch_size : int
               amount of new samples using in this updating step
                   constant, if batchtype = 'lin'
                   exponentially growing, if batchtype = 'exp'
                   1, if batchtype = 'none'
           m : int
               updating step, if batchtype = 'none' then m = t

           """


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

        self.statistics_decisions.evaluate_scores(scores_test)

        return dec, pdec

    def _get_mu(self, m):
        """
        Return the current value of mu_m
        m = t if batchtype = 'none', otherwise m  = batch updating round

        """
        # a = 1.0/(4)
        # b = np.sqrt(np.log(16.0*(self.t**2)*self.B.N/self.delta)/float(4*self.t))


        a = self.mu
        if m == 0:
            b = np.inf
        else:
            b = self.mu * np.sqrt(2) / np.sqrt(m)
        c = np.min([a, b])
        return np.min([1, c])


    def _solve_op(self, H, T1, m, best_pi, psi):

        """
        coordinate descent algorithm
        returns distribution Q

        Parameters
        ----------
        H : pd.DataFrame
            history H_t with all samples seen so until time t (phase 1 and phase2)
        T1: int
            size of phase 1
        m : int
            updating step, if batchtype = 'none' then m = t
        best_pi : RegressionPolicy
            best policy returned by fair oracle
        psi: float
            regularization parameter for regret b_t(pi)
        batch_size : int
            amount of new samples using in this updating step
                constant, if batchtype = 'lin'
                exponentially growing, if batchtype = 'exp'
                1, if batchtype = 'none'

        """

        num_policies_added = 0


        mu = self._get_mu(m)

        t = H.shape[0]

        #  initialize Q = 0
        Q = []  ## warm start: self.weights

        predictions = {}
        q_losses = {}

        leader_loss, predictions = self.get_cum_loss(H, best_pi, predictions)

        iterations = 0

        # loop
        while iterations < int(self.num_iterations):
            iterations += 1

            score = np.sum([item[1] * (4 + ((q_losses[item[0]] - leader_loss) / (psi * t * mu))) for item in Q])

            # first constraint (regret)
            if score > 4:
                #  if violated, shrink all weights with c<1
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

            # smoothed distribution over decisions
            q = (1.0 - 2 * mu) * q + (mu)

            # calculate v(d), s(d)
            v = 1.0 / (t * q)
            s = 1.0 / (t * (q ** 2))

            loss = H.loc[:,['l0', 'l1']].to_numpy()

            # reg(tilde) = max{Reg, 0}
            reg = loss - leader_loss
            reg[reg < 0] = 0
            bt = reg / (t * psi * mu)


            _H = H.loc[:, ['features', 'sensitive_features', 'label']]

            # D(Q, pi) for calling the MINIMIZATION oracle
            loss1 = pd.DataFrame(bt - v, columns=['l0', 'l1'], index=H.index)
            Dpimin_dataset = pd.concat([_H, loss1], axis=1)

            # D(Q, pi) for calculation when adding policy to Q
            loss2 = pd.DataFrame(v - bt, columns=['l0', 'l1'], index=H.index)
            Dpinormal_dataset = pd.concat([_H, loss2], axis=1)

            Dpimin_dataset1 = Dpimin_dataset.iloc[0:T1,:]
            Dpimin_dataset2 = Dpimin_dataset.iloc[T1:,:].drop(columns=['label'])

            #  calling oracle to obtain policy that violates variance constraint maximally
            pi = Argmin.argmin(self.randomthresholds, self.eps, self.nu, self.fairness, Dpimin_dataset1, Dpimin_dataset2)

            # caching of decisions by pi on history set
            if pi not in q_losses.keys():
                pi_loss, _ = self.get_cum_loss(H, pi, predictions)
                q_losses[pi] = pi_loss

            assert pi in predictions.keys(), "Uncached predictions for new policy pi"


            Dpi, _ = self.get_cum_loss(Dpinormal_dataset, pi, predictions)


            Dpi = Dpi - 4

            # Second constraint : variance constraint
            if Dpi > 0:
                loss3 = pd.DataFrame(v, columns=['l0', 'l1'], index=H.index)
                Vpi_dataset = pd.concat([_H, loss3], axis=1)

                loss4 = pd.DataFrame(s, columns=['l0', 'l1'], index=H.index)
                Spi_dataset = pd.concat([_H, loss4], axis=1)

                Vpi, _ = self.get_cum_loss(Vpi_dataset, pi, predictions)
                Spi, _ = self.get_cum_loss(Spi_dataset, pi, predictions)

                toadd = (Vpi + Dpi) / (2 * (1 - 2*mu) * Spi)

                #  add a new pi to Q, if constraint violated
                Q.append((pi, toadd))
                num_policies_added +=1

            else:
                # naturally end algorithm, when constraint not violated anymore
                self.list_num_policies_added.append(num_policies_added)
                return Q

        # end algorithm, when maximum number of loops reached
        self.list_num_policies_added.append(num_policies_added)
        return Q

    def get_cum_loss(self, dataset, pi, predictions):

        """
        calculate cumulative loss for policy pi on dataset

        Parameters
        ----------
        dataset : pd.DataFrame
            dataset with loss to be calculated
        pi: RegressionPolicy
            size of phase 1
        predictions : dict
            cashed decisions from for policies

        """

        if pi not in predictions.keys():
            values = pi.get_all_decisions(dataset.loc[:, ['features', 'sensitive_features']])
            predictions[pi] = values

        prob = predictions[pi][1]
        dec = predictions[pi][0]
        loss_df = dataset.loc[:,['l0', 'l1']]

        loss_df.columns = range(loss_df.shape[1])
        loss = pd.Series(loss_df.lookup(loss_df.index, dec))
        score = loss.dot(prob)
        return score, predictions


