from collections import namedtuple

import numpy as np
from scipy.stats import rankdata

CucconiResult = namedtuple('CucconiResult', ('statistic', 'pvalue'))
CucconiMultisampleResult = namedtuple('CucconiMultisampleResult', ('statistic', 'pvalue'))


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
    """
    Method to perform a Cucconi scale-location test.
    Args:
        a (np.ndarray): vector of observations
        b (np.ndarray): vector of observations
        method (str): method for determining p-value,
            possible values are 'bootstrap' and 'permutation'
        replications (int): number of bootstrap replications
        ties (str): string specifying a method to deal with ties in data,
            possible values as for scipy.stats.rankdata

    Returns:
        tuple: namedtuple with test statistic value and the p-value

    Raises:
        ValueError: if 'method' parameter is not specified to 'bootstrap' or 'permutation'

    Examples:
        >>> np.random.seed(987654321) # set random seed to get the same result
        >>> sample_a = sample_b = np.random.normal(loc=0, scale=1, size=100)
        >>> cucconi_test(sample_a, sample_b, replications=10000)
        CucconiResult(statistic=3.7763314663244195e-08, pvalue=1.0)

        >>> np.random.seed(987654321)
        >>> sample_a = np.random.normal(loc=0, scale=1, size=100)
        >>> sample_b = np.random.normal(loc=10, scale=10, size=100)
        >>> cucconi_test(sample_a, sample_b, method='permutation')
        CucconiResult(statistic=2.62372293956099, pvalue=0.000999000999000999)

    """
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

    p_value = (len(np.array(h0_distribution)[h0_distribution >= test_statistics]) + 1) / (replications + 1)

    return CucconiResult(statistic=test_statistics, pvalue=p_value)


def _cucconi_multisample_test_statistic(samples, ties='average'):
    lengths = np.cumsum([0] + [s.shape[0] for s in samples])
    ranked_data = rankdata(np.concatenate(samples), method=ties)
    samples_ranks = [ranked_data[lengths[k]:lengths[k+1]] for k, _ in enumerate(lengths[:-1])]

    n_i = np.array([s.shape[0] for s in samples])
    n = sum(n_i)

    expected_values = n_i * (n + 1) * (2 * n + 1) / 6
    std_deviations = np.sqrt(n_i * (n - n_i) * (n + 1) * (2 * n + 1) * (8 * n + 11) / 180)
    correlation = -(30*n+14*n**2+19)/((8*n+11)*(2*n+1))

    U = np.array([(np.sum(sample**2) - expected_values[i])/std_deviations[i] for i, sample in enumerate(samples_ranks)])
    V = np.array([(np.sum((n+1-sample)**2) - expected_values[i])/std_deviations[i] for i, sample in enumerate(samples_ranks)])
    MC = np.mean(U**2+V**2-2*U*V*correlation)/(2-2*correlation**2)

    return MC


def _cucconi_multisample_dist_bootstrap(samples, replications=1000, ties='average'):
    lengths = [len(s) for s in samples]
    h0_data = np.concatenate(samples)

    def bootstrap_test(replication_index):
        new_samples = [np.random.choice(h0_data, size=n, replace=True) for n in lengths]
        return _cucconi_multisample_test_statistic(samples=new_samples, ties=ties)

    return sorted(map(bootstrap_test, range(replications)))


def _cucconi_multisample_dist_permutation(samples, replications=1000, ties='average'):
    lengths = lengths = np.cumsum([0] + [s.shape[0] for s in samples])
    h0_data = np.concatenate(samples)

    def permuted_test(replication_index):
        permuted_data = np.random.permutation(h0_data)
        new_samples = [permuted_data[lengths[k]:lengths[k + 1]] for k, _ in enumerate(lengths[:-1])]
        return _cucconi_multisample_test_statistic(samples=new_samples, ties=ties)

    return sorted(map(permuted_test, range(replications)))


def cucconi_multisample_test(samples, method='bootstrap', replications=1000, ties='average'):
    """
    Method to perform a multisample Cucconi scale-location test.
    Args:
        samples (List[numpy.ndarray]): list of observation vectors
        method (str): method for determining p-value,
            possible values are 'bootstrap' and 'permutation'
        replications (int): number of bootstrap replications
        ties (str): string specifying a method to deal with ties in data,
            possible values as for scipy.stats.rankdata

    Returns:
        tuple: namedtuple with test statistic value and the p-value

    Raises:
        ValueError: if 'method' parameter is not specified to 'bootstrap' or 'permutation'

    Examples:
        >>> np.random.seed(987654321) # set random seed to get the same result
        >>> sample_a = sample_b = np.random.normal(loc=0, scale=1, size=100)
        >>> cucconi_multisample_test([sample_a, sample_b], replications=100000)
        CucconiMultisampleResult(statistic=6.996968353551774e-07, pvalue=1.0)

        >>> np.random.seed(987654321)
        >>> sample_a = np.random.normal(loc=0, scale=1, size=100)
        >>> sample_b = np.random.normal(loc=10, scale=10, size=100)
        >>> cucconi_multisample_test([sample_a, sample_a, sample_b], method='permutation')
        CucconiMultisampleResult(statistic=45.3891929069273, pvalue=0.000999000999000999)

    """
    assert isinstance(samples, list), 'Sample must be provided in a form of list, ' \
                                      'but were provided in {} format.'.format(type(samples))
    samples = list(map(np.asarray, samples))

    test_statistics = _cucconi_multisample_test_statistic(samples=samples, ties=ties)

    if method == 'permutation':
        h0_distribution = _cucconi_multisample_dist_bootstrap(samples=samples, replications=replications, ties=ties)
    elif method == 'bootstrap':
        h0_distribution = _cucconi_multisample_dist_permutation(samples=samples, replications=replications, ties=ties)
    else:
        raise ValueError(
            "Unknown method for constructing the distribution, "
            "possible values are ['bootstrap', 'permutation'], but {} was provided".format(method))

    p_value = (len(np.array(h0_distribution)[h0_distribution >= test_statistics]) + 1) / (replications + 1)

    return CucconiMultisampleResult(statistic=test_statistics, pvalue=p_value)
