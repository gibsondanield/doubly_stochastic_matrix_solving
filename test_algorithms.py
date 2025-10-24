import unittest

from main import *


class MyTestCase(unittest.TestCase):
    def test_make_empty(self):
        self.assertEqual(True, make_empty(4, 4).isnull().all().all())  # add assertion here

    def test_make_full(self):
        np.testing.assert_array_equal(make_full(4, 2), [[0, 0], [0, 1], [1, 0], [1, 1]])

    def test_make_full1(self):
        np.testing.assert_array_equal(make_full(4, 2, 1), [[0, 0], [0, 1], [1, 0], [1, 1]])

    def test_fill_from_constraints2(self):
        ... # make some more of these

    def test_fill_from_constraints(self):
        arr = make_full(4, 2)
        arr[0][0] = pd.NA
        np.testing.assert_array_equal(fill_from_constraints([2, 2], arr), make_full(4, 2))

    def test_entropy_full(self):
        for n in range(2, 10):
            self.assertEqual(compute_matrix_entropy(make_full(2 ** n, n)), 0)

    def test_entropy_naive(self):
        for n in range(2, 10):
            self.assertEqual(compute_matrix_entropy(make_empty(2 ** n, n).astype(float).fillna(.5)), n * 2 ** n)

    def test_entropy(self):
        self.assertEqual(entropy(.5), 1)

    def test_random(self):
        # make sure random completes in n cycles
        # make sure its less than the worst case scenario
            detectors = 3
            bottles = 2 ** 3
            self.assertEqual(next(iter(run([Random], detectors, bottles).values())),
                               bottles)
        # test all permutations?

    def test_random_with_constraints(self):
        # make sure its less than the worst case scenario
        detectors=3
        bottles=2**3
        self.assertGreater(next(iter(run([RandomWithConstraintSolving],detectors, bottles).values())),   bottles-detectors)

    def test_solve_probability_matrix(self):
        p = pd.DataFrame(index=list(range(4)), columns=list(range(4)))
        p.iloc[0, 2] = 0
        p.iloc[0, 3] = 0
        ans = solve_probability_matrix(p)
        expected = [[.5, .5, 0, 0], [1/6,1/6,1/3,1/3], [1/6,1/6,1/3,1/3], [1/6,1/6,1/3,1/3]]
        np.testing.assert_array_almost_equal(ans.values, expected)
    def solve_overspecified(self):
        a=[
            [2, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 1],
        ]
        b = [2, 1, 2]
if __name__ == '__main__':
    unittest.main()
