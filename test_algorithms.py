import unittest

import numpy as np
import pandas as pd
import sympy

from inequality_solving import solve_vector_bounds
from main import Cellar, run
from algorithms import Random, RandomWithConstraintSolving
from matrix_utilities import make_empty, make_full, brute_force_probabilities, fill_from_constraints, \
    binary_to_probability_matrix, solve_probability_matrix, probability_matrix_to_binary_probabilities, entropy, \
    compute_matrix_entropy, exact_probability_matrix

# The LP only determines a cell when the constraints (plus symmetry) pin it down; elsewhere it returns a
# feasible but inexact fill-in. These brute-force comparisons document that limitation (see the blog
# post); exact_probability_matrix is the method that matches brute force in general.
LP_IS_APPROXIMATE = unittest.expectedFailure


def brute_force_vs_calculation_from_arr(arr):
    p1 = solve_probability_matrix(binary_to_probability_matrix(arr))
    p = brute_force_probabilities(arr)
    np.testing.assert_array_almost_equal(p, p1)


class TestAlgorithms(unittest.TestCase):
    def test_make_empty(self):
        self.assertEqual(True, make_empty(4, 4).isnull().all().all())

    def test_make_full(self):
        np.testing.assert_array_equal(make_full(4, 2), [[0, 0],
                                                        [1, 0],
                                                        [0, 1],
                                                        [1, 1]])

    def test_make_full1(self):
        np.testing.assert_array_equal(make_full(4, 2, 1), [[0, 0],
                                                           [1, 0],
                                                           [1, 1],
                                                           [0, 1]])

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
        self.assertEqual(1, entropy(.5))

    def test_random(self):
        # make sure random completes in n cycles
        # make sure its less than the worst case scenario
        detectors = 3
        bottles = 2 ** 3
        self.assertEqual(bottles * detectors, next(iter(run([Random], detectors, bottles).values())))

    # test all permutations?

    def test_random_with_constraints(self):
        # make sure its less than the worst case scenario which is where each column gets only one for free
        detectors = 3
        bottles = 2 ** 3
        self.assertGreater(next(iter(run([RandomWithConstraintSolving], detectors, bottles).values())),
                           bottles - detectors)

    def test_solve_probability_matrix(self):
        p = pd.DataFrame(index=list(range(4)), columns=list(range(4)))
        p.iloc[0, 2] = 0
        p.iloc[0, 3] = 0
        ans = solve_probability_matrix(p)
        expected = [[.5, .5, 0, 0],
                    [1 / 6, 1 / 6, 1 / 3, 1 / 3],
                    [1 / 6, 1 / 6, 1 / 3, 1 / 3],
                    [1 / 6, 1 / 6, 1 / 3, 1 / 3]]
        np.testing.assert_array_almost_equal(ans.values, expected)

    def test_brute_force_probabilities(self):
        arr = make_empty(4, 2)
        arr.loc[0, 1] = False
        p = brute_force_probabilities(arr)
        expected = [[.5, .5, 0, 0],
                    [1 / 6, 1 / 6, 1 / 3, 1 / 3],
                    [1 / 6, 1 / 6, 1 / 3, 1 / 3],
                    [1 / 6, 1 / 6, 1 / 3, 1 / 3]]
        np.testing.assert_array_almost_equal(p.values, expected)

    @LP_IS_APPROXIMATE
    def test_brute_force_probabilities2(self):
        arr = make_empty(8, 3)
        arr.loc[0, 1] = False
        arr.loc[1, 2] = False
        brute_force_vs_calculation_from_arr(arr)

    def test_brute_force_probabilities3(self):
        # can be solved with constraints
        arr = make_empty(8, 3)
        arr.loc[0, 1] = False
        arr.loc[0, 2] = False
        arr.loc[1, 1] = False
        arr.loc[1, 2] = False
        arr.loc[2, 1] = False
        # automorphism makes this unsolvable without increasing rank
        brute_force_vs_calculation_from_arr(arr)

    @LP_IS_APPROXIMATE
    def test_brute_force_probabilities4(self):
        arr = make_empty(8, 3)
        arr.loc[0, 1] = False
        arr.loc[1, 2] = False
        arr.loc[2, 0] = False
        brute_force_vs_calculation_from_arr(arr)

    @LP_IS_APPROXIMATE
    def test_brute_force_probabilities5(self):
        arr = make_empty(8, 3)
        arr.loc[0, 1] = False
        arr.loc[1, 2] = False
        arr.loc[2, 0] = False
        arr.loc[3, 0] = False
        arr.loc[4, 1] = False
        arr.loc[5, 2] = False
        brute_force_vs_calculation_from_arr(arr)

    @LP_IS_APPROXIMATE
    def test_brute_force_probabilities6(self):
        arr = make_empty(8, 3)
        arr.loc[0, 1] = False
        arr.loc[1, 2] = False
        arr.loc[2, 0] = False
        arr.loc[3, 0] = False
        arr.loc[4, 1] = False
        arr.loc[5, 2] = False
        arr.loc[6, 0] = True
        arr.loc[7, 1] = True
        brute_force_vs_calculation_from_arr(arr)

    def test_exact_probabilities_match_brute_force(self):
        cases = [{(0, 1): False, (1, 2): False},
                 {(0, 1): False, (0, 2): False, (1, 1): False, (1, 2): False, (2, 1): False},
                 {(0, 1): False, (1, 2): False, (2, 0): False},
                 {(0, 1): False, (1, 2): False, (2, 0): False, (3, 0): False, (4, 1): False, (5, 2): False},
                 {(0, 1): False, (1, 2): False, (2, 0): False, (3, 0): False, (4, 1): False, (5, 2): False,
                  (6, 0): True, (7, 1): True}]
        for obs in cases:
            arr = make_empty(8, 3)
            for (r, c), v in obs.items():
                arr.loc[r, c] = v
            np.testing.assert_array_almost_equal(exact_probability_matrix(binary_to_probability_matrix(arr)),
                                                 brute_force_probabilities(arr).astype(float))

    def test_probability_matrix_to_binary_probabilities(self):
        p = pd.DataFrame([[.5, .5, 0, 0],
                          [1 / 6, 1 / 6, 1 / 3, 1 / 3],
                          [1 / 6, 1 / 6, 1 / 3, 1 / 3],
                          [1 / 6, 1 / 6, 1 / 3, 1 / 3]])
        binary = probability_matrix_to_binary_probabilities(p, 2)
        # LSB-first (same convention as make_full): bit 0 = p[:,1] + p[:,3], bit 1 = p[:,2] + p[:,3]
        expected = pd.DataFrame({0: {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5}, 1: {0: 0.0, 1: 0.6666666666666666, 2: 0.6666666666666666, 3: 0.6666666666666666}})
        pd.testing.assert_frame_equal(binary, expected)



    def test_bit_order_round_trip(self):
        '''decimal_to_binary_arr must use the same (LSB-first) bit order as make_full, otherwise
        bits inferred from the probability matrix land in the wrong detector column'''
        from matrix_utilities import decimal_to_binary_arr
        for n in range(1, 6):
            full = make_full(2 ** n, n).astype(bool)
            for i in range(2 ** n):
                np.testing.assert_array_equal(decimal_to_binary_arr(i, n), full.loc[i].values)
        # a known shuffled cellar -> permutation matrix -> bit probabilities gives the cellar back
        n = 3
        cellar = Cellar(n, seed=7).arr.reset_index(drop=True)
        codes = cellar.astype(int).values @ (1 << np.arange(n))
        p = pd.DataFrame(np.eye(2 ** n)[codes])
        np.testing.assert_array_equal(probability_matrix_to_binary_probabilities(p, n).values, cellar.values)

    def test_binary_to_probability_matrix(self):
        arr = pd.DataFrame([[None, True], [None, None], [None, None], [None, None]])
        pd.testing.assert_frame_equal(binary_to_probability_matrix(arr), pd.DataFrame(([[0, 0, np.nan, np.nan],
                                                                                    [np.nan, np.nan, np.nan, np.nan],
                                                                                    [np.nan, np.nan, np.nan, np.nan],
                                                                                    [np.nan, np.nan, np.nan, np.nan]]),
                                                                                  dtype=object))


    def test_solve_probability_matrix1(self):
        p = pd.DataFrame.from_dict(
            {0: {0: np.nan, 1: 0, 2: np.nan, 3: np.nan, 4: np.nan, 5: np.nan, 6: np.nan, 7: np.nan},
             1: {0: 0, 1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan, 5: np.nan, 6: np.nan, 7: np.nan},
             2: {0: np.nan, 1: 0, 2: np.nan, 3: 0, 4: np.nan, 5: np.nan, 6: np.nan, 7: np.nan},
             3: {0: 0, 1: np.nan, 2: np.nan, 3: 0, 4: np.nan, 5: np.nan, 6: np.nan, 7: np.nan},
             4: {0: np.nan, 1: 0, 2: np.nan, 3: np.nan, 4: np.nan, 5: np.nan, 6: np.nan, 7: np.nan},
             5: {0: 0, 1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan, 5: np.nan, 6: np.nan, 7: np.nan},
             6: {0: np.nan, 1: 0, 2: np.nan, 3: 0, 4: np.nan, 5: np.nan, 6: np.nan, 7: np.nan},
             7: {0: 0, 1: np.nan, 2: np.nan, 3: 0, 4: np.nan, 5: np.nan, 6: np.nan, 7: np.nan}}
            )
        solve_probability_matrix(p)

    def test_solve_probability_matrix2(self):
        p = pd.DataFrame.from_dict(
            {0: {0: np.nan, 1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan, 5: np.nan, 6: np.nan, 7: np.nan},
             1: {0: 0, 1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan, 5: np.nan, 6: np.nan, 7: np.nan},
             2: {0: np.nan, 1: np.nan, 2: np.nan, 3: 0, 4: np.nan, 5: np.nan, 6: np.nan, 7: np.nan},
             3: {0: 0, 1: np.nan, 2: np.nan, 3: 0, 4: np.nan, 5: np.nan, 6: np.nan, 7: np.nan},
             4: {0: np.nan, 1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan, 5: np.nan, 6: np.nan, 7: np.nan},
             5: {0: 0, 1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan, 5: np.nan, 6: np.nan, 7: np.nan},
             6: {0: np.nan, 1: np.nan, 2: np.nan, 3: 0, 4: np.nan, 5: np.nan, 6: np.nan, 7: np.nan},
             7: {0: 0, 1: np.nan, 2: np.nan, 3: 0, 4: np.nan, 5: np.nan, 6: np.nan, 7: np.nan}})
        solve_probability_matrix(p)


    def test_solve_inequalities1(self):
        tau0 = sympy.symbols('tau0', real=True)
        x = sympy.Tuple(2, tau0 + 1, -tau0, 4, 1 - tau0, tau0)
        upper = (2, 1, 1, 5, 2, 2)
        sol, syms, x_eval = solve_vector_bounds(x, upper)
        expected = [2, 1, 0, 4, 1, 0]
        np.testing.assert_array_almost_equal(x_eval, expected)


if __name__ == '__main__':
    unittest.main()
