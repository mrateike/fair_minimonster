from src.contextual_bandit.Policy import *
from src.fairlearn.reductions._exponentiated_gradient.exponentiated_gradient import ExponentiatedGradient
from sklearn.linear_model import LogisticRegression
_L0 = "l0"
_L1 = "l1"


"""
This class calls the fair orcale for the contextual bandit algorithm
"""


def argmin(c, eps, nu, fairness, dataset1, dataset2=None):
    """
    Calls the fair oracle to return the best fair policy
    Args:
            c: np.array
                array of random thresholds for decision making
            eps: float
                fairness constraint slack variable (relaxation parameter)
            nu: float
                accuracy parameter
            fairness: str
                type of fairness to optimize for
            dataset1: pd.DataFrame
                phase 1 dataset (T1)
            dataset2: pd.DataFrame
                phase 2 dataset

    Imports:
            Logistic Regression from scikit_learn
                An estimator implementing methods :code:`fit(X, y, sample_weight)` and
                :code:`predict(X)`, where `X` is the matrix of features, `y` is the
                vector of labels, and `sample_weight` is a vector of weights;
                labels `y` and predictions returned by :code:`predict(X)` are either
                0 or 1.
    Returns:
            pi: RegressionPolicy
                Exponentiated gradient estimator (stochastic policy)
    """


    estimator = LogisticRegression(solver='liblinear', fit_intercept=True)

    expgrad_XA = ExponentiatedGradient(c,
        dataset1,
        estimator,
        constraints=fairness,
        eps=eps,
        nu=nu)

    # Option for calling the oracle on phase 1 data only
    if dataset2.empty:
        expgrad_XA.fit()

    # Option for calling the oracle on phase 2 data only
    else:
        # sensitive and non-sensitive features as input for prediction
        XA = dataset2.loc[:,['features', 'sensitive_features']]
        # sensitive features for fairness evaluation
        A = dataset2.loc[:, 'sensitive_features']
        # loss vector for binary decisions d in {0,1}
        L = dataset2.loc[:,['l0', 'l1']]

        expgrad_XA.fit(
            XA,
            L,
            sensitive_features=A)

    # Returning policy as ResgressionPolicy type
    pi = RegressionPolicy(expgrad_XA)
    return pi


