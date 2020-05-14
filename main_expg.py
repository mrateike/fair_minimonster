from fairlearn.reductions._exponentiated_gradient.run_expg import RunExpg
from fairlearn.reductions._moments.conditional_selection_rate import DemographicParity
import numpy as np
n1_train = 100
n2_train = 16000
n_test = 5000
shifts = True


acc = []
DP = []
TPR = []
FPR = []
EO = []

fairness = DemographicParity()

T = 1
i = 0
for i in range(i, T):
    print('I am running')
    improved_acc, improved_DP, improved_TPR, improved_FPR, improved_equalOdds \
        = RunExpg.play(n1_train, n2_train, n_test, shifts, fairness)

    acc.append(improved_acc)
    DP.append(improved_DP)
    TPR.append(improved_TPR)
    FPR.append(improved_FPR)
    EO.append(improved_equalOdds)
    i+=1

print('ACC mean', np.mean(acc))
print("DP mean", np.mean(DP))

print('ACC var', np.var(acc))
print("DP var", np.var(DP))