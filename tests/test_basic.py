import unittest

from nonparstat.Cucconi import *


class Cucconi(unittest.TestCase):
    def test_equal(self):
        sample_a = sample_b = np.random.normal(loc=0, scale=1, size=100)
        self.assertGreater(cucconi_test(sample_a, sample_b).pvalue, 0.99)

    def test_mean(self):
        sample_a = np.random.normal(loc=0, scale=1, size=100)
        sample_b = np.random.normal(loc=10, scale=1, size=100)
        self.assertLess(cucconi_test(sample_a, sample_b).pvalue, 0.001)

    def test_variance(self):
        sample_a = np.random.normal(loc=0, scale=1, size=100)
        sample_b = np.random.normal(loc=0, scale=10, size=100)
        self.assertLess(cucconi_test(sample_a, sample_b).pvalue, 0.001)

    def test_mean_variance(self):
        sample_a = np.random.normal(loc=0, scale=1, size=100)
        sample_b = np.random.normal(loc=10, scale=10, size=100)
        self.assertLess(cucconi_test(sample_a, sample_b).pvalue, 0.001)

    def test_equal_permutations(self):
        sample_a = sample_b = np.random.normal(loc=0, scale=1, size=100)
        self.assertGreater(cucconi_test(sample_a, sample_b, method='permutation').pvalue, 0.99)

    def test_mean_permutations(self):
        sample_a = np.random.normal(loc=0, scale=1, size=100)
        sample_b = np.random.normal(loc=10, scale=1, size=100)
        self.assertLess(cucconi_test(sample_a, sample_b, method='permutation').pvalue, 0.001)

    def test_variance_permutations(self):
        sample_a = np.random.normal(loc=0, scale=1, size=100)
        sample_b = np.random.normal(loc=0, scale=10, size=100)
        self.assertLess(cucconi_test(sample_a, sample_b, method='permutation').pvalue, 0.001)

    def test_mean_variance_permutations(self):
        sample_a = np.random.normal(loc=0, scale=1, size=100)
        sample_b = np.random.normal(loc=10, scale=10, size=100)
        self.assertLess(cucconi_test(sample_a, sample_b, method='permutation').pvalue, 0.001)

    def test_method(self):
        sample_a = sample_b = np.random.normal(loc=0, scale=1, size=100)
        self.assertRaises(ValueError, cucconi_test, sample_a, sample_b, method='exact')


if __name__ == '__main__':
    unittest.main()
