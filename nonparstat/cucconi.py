import numpy as np
from scipy.stats import rankdata
from collections import namedtuple

CucconiResult = namedtuple('CucconiResult', ('statistic', 'pvalue'))


def _cucconi_test_statistic(a, b, ties='average'):
    n1 = len(a)
    n2 = len(b)
    n = n1 + n2
    alldata = np.concatenate((a, b))
    ranked = rankdata(alldata, method=ties)
    a_ranks = ranked[:n1]

    rho = 2 * (n ** 2 - 4) / ((2 * n + 1) * (8 * n + 11)) - 1
    U = (6 * np.sum(np.square(a_ranks)) - n1 * (n + 1) * (2 * n + 1)) / np.sqrt(
        n1 * n2 * (n + 1) * (2 * n + 1) * (8 * n + 11) / 5)
    V = (6 * np.sum(np.square(n + 1 - a_ranks)) - n1 * (n + 1) * (2 * n + 1)) / np.sqrt(
        n1 * n2 * (n + 1) * (2 * n + 1) * (8 * n + 11) / 5)
    C = (U ** 2 + V ** 2 - 2 * rho * U * V) / 2 * (1 - rho ** 2)

    return C


def _cucconi_dist_permutation(a, b, replications=1000, ties='average'):
    n1 = len(a)
    n2 = len(b)
    h0_data = np.concatenate([a, b])

    def permuted_test(replication_index):
        permuted_data = np.random.permutation(h0_data)
        new_a = permuted_data[:n1]
        new_b = permuted_data[n1:]
        return _cucconi_test_statistic(a=new_a, b=new_b, ties=ties)

    return sorted(map(permuted_test, range(replications)))


def _cucconi_dist_bootstrap(a, b, replications=1000, ties='average'):
    n1 = len(a)
    n2 = len(b)
    h0_data = np.concatenate([a, b])

    def bootstrap_test(replication_index):
        new_a = np.random.choice(h0_data, size=n1, replace=True)
        new_b = np.random.choice(h0_data, size=n2, replace=True)
        return _cucconi_test_statistic(new_a, new_b, ties=ties)

    return sorted(map(bootstrap_test, range(replications)))


def cucconi_test(a, b, method='bootstrap', replications=1000, ties='average'):
    a, b = map(np.asarray, (a, b))

    test_statistics = _cucconi_test_statistic(a=a, b=b, ties=ties)

    if method == 'permutation':
        h0_distribution = _cucconi_dist_permutation(a=a, b=b, replications=replications, ties=ties)
    elif method == 'bootstrap':
        h0_distribution = _cucconi_dist_bootstrap(a=a, b=b, replications=replications, ties=ties)
    else:
        raise ValueError(
            "Unknown method for constructing the distribution, "
            "possible values are ['bootstrap', 'permutation'], but {} was provided".format(method))

    p_value = (len(np.array(h0_distribution)[h0_distribution >= test_statistics]) + 1)/(replications + 1)

    return CucconiResult(statistic=test_statistics, pvalue=p_value)
