import numpy as np
from scipy.stats import rankdata, f
from collections import namedtuple

Podgor_GastwirthResult = namedtuple('Podgor_GastwirthResult', ('statistic', 'pvalue'))


def _podgor_gastwirth_test_statistic(a, b, ties='average'):
    n1 = len(a)
    n2 = len(b)
    n = n1 + n2
    alldata = np.concatenate((a, b))
    ranked = rankdata(alldata, method=ties)
    ranked_sq = np.square(ranked)
    I_vector = np.hstack([np.repeat(1, n1), np.repeat(0, n2)])
    S_matrix = np.vstack([np.repeat(1, n), ranked, ranked_sq])
    b_vector = np.matmul(np.linalg.inv(np.matmul(S_matrix, S_matrix.T)), np.matmul(S_matrix, I_vector))
    numerator = (np.matmul(b_vector.T, np.matmul(S_matrix, I_vector)) - n1 ** 2 / n) / 2
    denumerator = (n1 - np.matmul(b_vector.T, np.matmul(S_matrix, I_vector))) / (n - 3)
    return numerator / denumerator


def _podgor_gastwirth_dist(a, b, x):
    df1 = 2
    df2 = len(a) + len(b) - 3
    return f.cdf(x=x, dfn=df1, dfd=df2)


def podgor_gastwirth_test(a, b, ties='average'):
    a, b = map(np.asarray, (a, b))

    test_statistics = _podgor_gastwirth_test_statistic(a, b, ties=ties)

    p_value = 1 - _podgor_gastwirth_dist(a, b, test_statistics)

    return Podgor_GastwirthResult(statistic=test_statistics, pvalue=p_value)
