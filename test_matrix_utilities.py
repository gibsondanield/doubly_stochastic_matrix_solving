import unittest
from matrix_utilities import *

class MyTestCase(unittest.TestCase):

    def test_3x3_all_ones(self):
        """Test 3x3 all-ones matrix - bound should equal actual permanent of 6"""
        df = pd.DataFrame([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ])
        bound = permanent_upper_bound(df)
        self.assertAlmostEqual(bound, 6.0, places=4)

    def test_4x4_identity(self):
        """Test 4x4 identity matrix - permanent is 1, bound should be 1"""
        df = pd.DataFrame(np.eye(4, dtype=int))
        bound = permanent_upper_bound(df)
        self.assertAlmostEqual(bound, 1.0, places=4)

    def test_2x2_all_ones(self):
        """Test 2x2 all-ones matrix - permanent is 2, bound should be 2"""
        df = pd.DataFrame([
            [1, 1],
            [1, 1]
        ])
        bound = permanent_upper_bound(df)
        self.assertAlmostEqual(bound, 2.0, places=4)

    def test_4x4_sparse_matrix(self):
        """Test 4x4 sparse matrix"""
        df = pd.DataFrame([
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1]
        ])
        bound = permanent_upper_bound(df)
        # Bound should be finite and positive
        self.assertGreater(bound, 0)
        self.assertLess(bound, 100)  # Reasonable upper limit
        # rows [2,2,2,2] -> (2!^(1/2))^4 = 4; cols [2,2,3,1] -> 2 * 6^(1/3) ~= 3.634; take the min
        self.assertAlmostEqual(bound, 2 * 6 ** (1 / 3), places=4)
        self.assertGreaterEqual(bound, 2)  # true permanent is 2

    def test_non_uniform_row_sums(self):
        """Test 3x3 matrix with varying row sums [1,2,3]"""
        df = pd.DataFrame([
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1]
        ])
        bound = permanent_upper_bound(df)
        # Row sums: [1,2,3] -> bound = 1 * sqrt(2) * cbrt(6)
        expected = 1 * (2 ** 0.5) * (6 ** (1 / 3))
        self.assertAlmostEqual(bound, expected, places=4)

    def test_bound_is_upper_bound(self):
        """Verify bound >= actual permanent for a known case"""
        # 3x3 all-ones has permanent = 6
        df = pd.DataFrame(np.ones((3, 3), dtype=int))
        bound = permanent_upper_bound(df)
        actual_permanent = 6  # Known value
        self.assertGreaterEqual(bound, actual_permanent - 0.0001)

    def test_matrix_with_zero_rows(self):
        """Test matrix with a zero row - permanent is 0, bound should handle gracefully"""
        df = pd.DataFrame([
            [1, 1, 1],
            [0, 0, 0],
            [1, 1, 1]
        ])
        bound = permanent_upper_bound(df)
        # With a zero row, the bound should be 0 (product includes 0-sum term)
        self.assertAlmostEqual(bound, 0.0, places=4)

    def test_symmetric_matrix(self):
        """Test that symmetric matrices give same bound from rows and columns"""
        df = pd.DataFrame([
            [1, 1, 0],
            [1, 1, 1],
            [0, 1, 1]
        ])
        row_bound = bregman_minc(df.T)  # Row sums
        col_bound = bregman_minc(df)  # Column sums
        overall_bound = permanent_upper_bound(df)
        # For symmetric matrix, row and col bounds should be equal
        self.assertAlmostEqual(row_bound, col_bound, places=4)
        self.assertAlmostEqual(overall_bound, row_bound, places=4)

    def test_approx_choices_cache(self):
        """Test that lru_cache is working"""
        # Clear cache first
        approx_choices.cache_clear()

        # First call
        result1 = approx_choices(3)
        info1 = approx_choices.cache_info()

        # Second call with same value
        result2 = approx_choices(3)
        info2 = approx_choices.cache_info()

        # Results should be identical
        self.assertEqual(result1, result2)
        # Cache hits should increase
        self.assertEqual(info2.hits, info1.hits + 1)

    def test_approx_choices_zero(self):
        """A line with no nonzeros has no choices -> factor 0 (so perm bound is 0)"""
        result = approx_choices(0)
        self.assertEqual(result, 0)


if __name__ == '__main__':
    unittest.main()
