import time
from pathlib import Path
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.contextual_bandit import Argmin
from src.contextual_bandit.Evaluation import Evaluation

import os

my_path = os.path.abspath(__file__)  # Figures out the absolute path for you in case your working directory moves around.


class MiniMonster(object):
    """
    Implementation of MiniMonster with a scikit_learn learning algorithm as the AMO.
    """

    def __init__(self, B, fairness, dataset1, eps, nu):
        self.B = B
        self.dataset1 = dataset1
        self.loss = []
        self.opt_loss = []
        self.history = pd.DataFrame()
        self.num_amo_calls = 0
        self.mu = 0.01
        self.fairness = fairness

        self.eps = eps
        self.nu = nu

        self.statistics = Evaluation()

        self.DP_dict = {}
        self.TPR_dict = {}
        self.FPR_dict = {}
        self.EO_dict = {}
        self.auc_dict = {}
        self.mean_pred_dict = {}
        self.t = 0

    def fit(self, T, batch):

        XA = pd.DataFrame()
        A = pd.Series(name='sensitive_features')
        L = pd.DataFrame(columns=['l0', 'l1'])

        l1 = []
        l2 = []

        t = 1
        m = 1
        predictions = {}
        Q = []
        best_pi = None

        # phase 2

        # print('--- EVALUATION -- first log policy')

        Y1 = self.dataset1.loc[:, 'label']
        XA1 = self.dataset1.drop(columns=['sensitive_features', 'label'])
        best_pi = LogisticRegression(solver='liblinear', fit_intercept=True)
        best_pi.fit(XA1, Y1)
        # print('Y1', Y1 )
        # print('XA1', XA1)
        self.statistics.evaluate(best_pi)


        while t < T + 1:
            # print('------- Phase 2: Time t = ', t, '--- new data point-----------')

            # DF, S, S
            xa, y, a = self.B.get_new_context()

            XA = XA.append(xa, ignore_index=True)
            A = A.append(a, ignore_index=True)


            d, p = self.sample(xa, Q, best_pi, m)
            #print('d', d)
            l = self.B.get_loss(d, y)

            # IPS loss, attention: order of columns might change
            # sometimes [l1, l0], sometimes [l0, l1]
            full_lvec = pd.DataFrame()
            # print('l', l)
            # print('p', p)
            full_lvec.at[0, d] = l[d] / p[d]
            full_lvec.at[0, 1 - d] = l[1 - d] / p[1 - d]
            full_lvec.rename(columns={0: 'l0', 1: 'l1'}, inplace=True)
            # print('full_lvec', full_lvec)
            L = L.append(full_lvec, ignore_index=True)

            # kind of batch formula defined here
            if batch == "exp":
                tau_m = 2 ** (m - 1)
            elif batch.startswith('lin'):
                #batchsize = [int(s) for s in batch.split() if s.isdigit()][0]
                batchsize = 500
                tau_m = batchsize * m
            elif batch == "none":
                tau_m = t
            else:
                print('error in defining batch')

            # ---------- update batch (only push new ones) --------
            if t == tau_m:
                print('--- batch update', m, 'at time', t)

                dataset2_batch = pd.concat([XA, L, A], axis=1)

                self.history = self.history.append(dataset2_batch, ignore_index=True)
                print('--- EVALUATION -- batch update: best policy')
                best_pi, log_Reg = Argmin.argmin(self.eps, self.nu, self.fairness, self.dataset1, self.history)
                self.statistics.evaluate(best_pi)
                print('--- EVALUATION -- comparison: logReg')
                self.statistics.evaluate(log_Reg)

                assert best_pi != None, 'no best_pi returned'
                self.num_amo_calls += 1

                Q, best_pi = self._solve_op(self.history, t, best_pi)

                XA = pd.DataFrame()
                A = pd.Series(name='sensitive_features')
                L = pd.DataFrame()

                m += 1

            # ----- evaluate loss -----------
            self.loss.append(l)
            l1.append(np.sum(self.loss) / t)

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

        # --- print values from Evaluation

        # plt.plot([1, 2, 3, 4])
        # plt.ylabel('some numbers')
        # plt.show()

        accuracy = self.statistics.acc_list_overall
        mean_pred = self.statistics.mean_pred_overall_list
        util = self.statistics.util_list
        DP = self.statistics.DP_list
        TPR = self.statistics.TPR_list
        EO = self.statistics.EO_list
        acc_dict = {0: self.statistics.acc_list_0, 1: self.statistics.acc_list_1,
                    'overall': self.statistics.acc_list_overall}
        pred_dict = {0: self.statistics.mean_pred_0_list, 1: self.statistics.mean_pred_1_list,
                     'overall': self.statistics.mean_pred_overall_list}
        # print('utility', util)
        data = {'ACC': acc_dict, 'mean_pred': pred_dict, 'Utility': util, 'DP': DP, 'TPR': TPR, 'EO': EO}

        base_save_path = Path.cwd() / 'results'
        Path(base_save_path).mkdir(parents=True, exist_ok=True)
        timestamp = time.gmtime()
        ts_folder = time.strftime("%Y-%m-%d-%H-%M-%S", timestamp)
        base_save_path = "{}/{}".format(base_save_path, ts_folder)
        Path(base_save_path).mkdir(parents=True, exist_ok=True)
        parameter_save_path = "{}/parameters.json".format(base_save_path)
        # print('data to print', data)
        Evaluation.save_dictionary(data, parameter_save_path)

        fig_train = plt.figure()
        # It's the arrangement of subgraphs within this graph. The first number is how many rows of subplots; the second number is how many columns of subplots; the third number is the subgraph you're talking about now. In this case, there's one row and one column of subgraphs (i.e. one subgraph) and the axes are talking about the first of them. Something like fig.add_subplot(3,2,5) would be the lower-left subplot in a grid of three rows and two columns
        ax1_train = fig_train.add_subplot(111)
        ax1_train.scatter(range(0, len(accuracy)), accuracy, label='accuracy')
        ax1_train.scatter(range(0, len(DP)), DP, label='DP')
        plt.xlabel("iterations")
        plt.ylabel("DP/accuracy")
        plt.title('Full Bandit Run')
        plt.legend()
        plt.savefig(base_save_path + '/plot.png')


        return l1, l2, Q, best_pi

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

        p = np.zeros(2)

        if Q == []:
            dec = best_pi.predict(x)
            # print('first pi dec',dec)
            # print('sample: Q', Q)
            # print('sample: best_pi get_decision', dec)
            # dec = np.random.choice(2, size=1)
            p[dec] = 1
            p[1 - dec] = 0

        else:
            for item in Q:
                pi = item[0]
                w = item[1]
                p[pi.get_decision(x)] += w

            ## Mix in leader
            dec = best_pi.get_decision(x)
            # print('sample: best_pi get_decision', dec)
            w = 1 - np.sum([q[1] for q in Q])
            p[dec] += w
            # print('sample: after mix in leader', p)

        # in paper:
        p = (1 - 2 * self._get_mu(t)) * p + self._get_mu(t)
        # in code: p = (1 - self._get_mu(t)) * p + self._get_mu(t) / 2
        ## Take decision
        # print('sample: after mix in mu', p)

        assert p[1] >= 0 and p[1] <= 1, 'probability needs to be [0,1]'
        dec = np.random.binomial(1, p[1])

        dec = int(dec)

        return dec, p

    def _get_mu(self, t):
        """
        Return the current value of mu_t
        """
        # a = 1.0/(4)
        # b = np.sqrt(np.log(16.0*(self.t**2)*self.B.N/self.delta)/float(4*self.t))
        # a = self.mu
        # b = self.mu * np.sqrt(2) / np.sqrt(t)
        # c = np.min([a, b])
        # return np.min([1, c])
        return 0.01

    def _solve_op(self, H, t, best_pi, Q=None):
        """
        Main optimization logic for MiniMonster.
        """
        print('----- BEGIN SOLVE OP ----------')
        H = H
        # print('_solve_op: H', H)
        mu = self._get_mu(t)
        # print('mu', mu)
        # print('_solve_op: mu', mu)
        Q = []  ## self.weights
        # ## Warm-starting Q = Q
        # psi set in code to 1
        psi = 1

        # ------ added by me
        t = t
        best_pi = best_pi

        # ------------------

        predictions = {}

        leader_loss, predictions = self.get_loss(H, best_pi, predictions, t)
        # print('_solve_op: leader_loss', leader_loss)
        # print('_solve_op: predictions', predictions)

        # only makes sense, if it is a warm start, otherwise Q is empty anyway
        q_losses = {}
        # for item in Q:
        #     pi = item[0]
        #     (tmp,predictions) = self.get_loss(H, pi, predictions, features=features)
        #     q_losses[pi] = tmp

        iterations = 0

        # normally loop here max 20 iterations
        updated = True
        while updated and iterations < 20:
            iterations += 1
            updated = False

            ## First IF statement, leader_reward needs to be IPS as well as q_rewards, deleted self.B.K*(leader_rew..)
            score = np.sum([x[1] * (4 + 2 * (q_losses[x[0]] - leader_loss) / (psi * t * mu)) for x in Q])

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

            # sum over all dataset t
            for i in range(t):
                x = pd.DataFrame(H.iloc[i, 0:2]).transpose()
                a = pd.Series(H.loc[i, 'sensitive_features'], name='sensitive_features')

                # 2 times because later 2 times
                loss = [2 * H.loc[i, 'l0'], 2 * H.loc[i, 'l1']]

                # q = self._marginalize(Q, x, pred)
                q = np.zeros(2, dtype=np.longfloat)
                for item in Q:
                    pi = item[0]
                    w = item[1]
                    q[predictions[pi][i]] += w
                q = (1.0 - mu) * q + (mu) / 2
                # print('MM: q', q)

                v = 1.0 / (t * q)
                l = [x / (t * psi * mu) for x in loss]
                s = 1.0 / (t * (q ** 2))

                # print('v', v)
                # print('l', l)
                # print('s', s)
                x.reset_index(drop=True, inplace=True)
                a.reset_index(drop=True, inplace=True)

                loss1 = pd.DataFrame([l - v], columns=['l0', 'l1'])
                loss1.reset_index(drop=True, inplace=True)
                d1 = pd.concat([x, loss1, a], axis=1)
                Dpimin_dataset = Dpimin_dataset.append(d1, ignore_index=True)

                loss2 = pd.DataFrame([v - l], columns=['l0', 'l1'])
                loss2.reset_index(drop=True, inplace=True)
                d2 = pd.concat([x, loss2, a], axis=1)
                Dpinormal_dataset = Dpinormal_dataset.append(d2, ignore_index=True)

                loss3 = pd.DataFrame([v], columns=['l0', 'l1'])
                loss3.reset_index(drop=True, inplace=True)
                v = pd.concat([x, loss3, a], axis=1)
                Vpi_dataset = Vpi_dataset.append(v, ignore_index=True)

                loss4 = pd.DataFrame([s], columns=['l0', 'l1'])
                loss4.reset_index(drop=True, inplace=True)
                s = pd.concat([x, loss4, a], axis=1)
                Spi_dataset = Spi_dataset.append(s, ignore_index=True)

            ## AMO call
            # print('----- AMO 2 ----------')
            # print('--- EVALUATION -- loop update: high variance')
            pi, _ = Argmin.argmin(self.eps, self.nu, self.fairness, self.dataset1, Dpimin_dataset)
            self.num_amo_calls += 1
            # print('----- END AMO 2 ----------')

            ## This is mostly to make sure we have the predictions cached for this new policy
            if pi not in q_losses.keys():
                (tmp, predictions) = self.get_loss(H, pi, predictions, t)
                q_losses[pi] = tmp
                if q_losses[pi] < leader_loss:
                    best_pi = pi
                    leader_loss = q_losses[pi]

            assert pi in predictions.keys(), "Uncached predictions for new policy pi"

            (Dpi, predictions) = self.get_loss(Dpinormal_dataset, pi, predictions, t)
            Dpi = Dpi - 4 + 2 * leader_loss / (psi * t * mu)

            if Dpi > 0:
                updated = True
                # print('---- Second Constraint: broken -----')
                # print('Vpi_dataset', Vpi_dataset)
                # print('Vpi_dataset.columns', Vpi_dataset.columns)
                # print('Vpi_dataset.index', Vpi_dataset.index)
                Vpi, ptwo = self.get_loss(Vpi_dataset, pi, predictions, t)

                # print('Spi_dataset', Spi_dataset)
                # print('Spi_dataset.columns', Spi_dataset.columns)
                # print('Spi_dataset.index', Spi_dataset.index)
                Spi, ptwo = self.get_loss(Spi_dataset, pi, predictions, t)

                toadd = (Vpi + Dpi) / (2 * (1 - mu) * Spi)
                Q.append((pi, toadd))
            # else:
            #     print('----- END SOLVE OP naturally ----------')
            #     return Q, best_pi

        print('----- END SOLVE OP ----------')
        return Q, best_pi

    def get_loss(self, dataset, pi, predictions, t):
        # H: DF, best_pi: Policy, pred: dict, t:int, feat:DF

        t = t

        if pi not in predictions.keys():
            ## This is going to go horribly wrong if dataset is not the right size
            assert len(dataset) == t, "If predictions not yet cached, dataset should have len = self.t"
            # predictions[pi] = dict(zip([i for i in dataset.index], pi.get_all_decisions([row for index, row in dataset.iloc[:,0:2].iterrows()])))
            predictions[pi] = dict(zip([i for i in dataset.index], pi.get_all_decisions(dataset.iloc[:, 0:2])))

        score = 0.0
        for i in dataset.index:
            x = i
            l = dataset.filter(items=['l0', 'l1']).iloc[i, :]
            l = l.rename({'l0': 0, 'l1': 1})

            pred = predictions[pi][x]
            score += l.loc[pred]

        return score, predictions
