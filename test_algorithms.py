import unittest
import numpy as np
import pandas as pd

from algorithms import *
from main import *

class MyTestCase(unittest.TestCase):
    def test_make_empty(self):
        self.assertEqual(True, make_empty(4,4).isnull().all().all())  # add assertion here

    def test_make_full(self):
        np.testing.assert_array_equal(make_full(4,2), [[0,0],[0,1],[1,0],[1,1]])
    def test_fill_from_constraints(self):
        arr  = make_full(4,2)
        arr[0][0] = pd.NA
        np.testing.assert_array_equal(fill_from_constraints([2,2], arr), make_full(4,2))

    def test_entropy_full(self):
        for n in range(2,10):
            self.assertEqual(compute_entropy(make_full(2**n, n)),0)
    def test_entropy_naive(self):
        for n in range(2,10):
            self.assertEqual(compute_entropy(make_empty(2**n, n).astype(float).fillna(1/n)),2**n)

if __name__ == '__main__':
    unittest.main()
