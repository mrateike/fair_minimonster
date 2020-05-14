
from contextual_bandit import Runtime
import argparse

# parser = argparse.ArgumentParser(description='Bechavods Fair Minimonster')
# parser.add_argument('T1', type=int, help='phase 1 time steps')
# parser.add_argument('T2', type=int, help='phase 2 time steps')
# parser.add_argument('TT', type=int, help='test set size')
# parser.add_argument('fairness', type=str, help='fairness: DP')
# parser.add_argument('batch', type=str, help='batch: exp, lin batchsize, none')
# args = parser.parse_args()
# T1 = args.T1
# T2 = args.T2
# TT = args.TT
# fairness = args.fairness
# batch = args.batch

# without args
T1 = 10
T2 = 1
TT = 10
fairness = 'DP'
batch = 'none'

# batch can take values 'exp', 'lin batchsize', 'none'

eps = 0.01
nu = 1e-6
print('Im running')
Runtime.play(T1, T2, TT, fairness, batch, eps, nu)
