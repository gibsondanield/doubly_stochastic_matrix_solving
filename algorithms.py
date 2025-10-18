import numpy as np
import pandas as pd
from abc import ABC


def make_empty(n_rows,n_cols):
    df = pd.DataFrame(np.full((n_rows, n_cols), pd.NA, dtype='object'), dtype='boolean')
    return df


def make_full(n_rows,n_cols):
    df = pd.DataFrame([[i // (2**n) % 2 for i in range(n_rows)] for n in range(n_cols)]).T
    return df


def fill_from_constraints(col_zeros, matrix):
    new_matrix = matrix.copy()
    # count each col with missing values
    cols = (matrix==0).sum() == col_zeros
    # if a column has the maximum number of zeros, the rest must be ones
    if cols.any():
        new_matrix[cols.index[cols]] = matrix[cols.index[cols]].fillna(True)
    # vice versa
    cols = (new_matrix == 1).sum() == np.subtract(new_matrix.shape[0], col_zeros)
    if cols.any():
        new_matrix[cols.index[cols]] = new_matrix[cols.index[cols]].fillna(False)
    # if a row has n-1 entries and matches a row that is full, infer the last bit
    rows = matrix.isnull().sum(axis=1) == 1

    # if there is a change, run again
    if (new_matrix!=matrix).all().all():
        return fill_from_constraints(col_zeros, new_matrix)
    return new_matrix



class MeasurementAlgorithmABC(ABC):

    def __init__(self, n_bottles, n_detectors):
        ...
    def next(self):
        '''returns bottle, detector for the next measurement'''
        ...

    def update(self, bottle:int, detector:int, measurement:int):
        '''doesnt return anything, just gives the last measurement to the algorithm'''
        ...

    def done(self):
        '''returns True when algortithm has solved everything'''
        ...

class Random(MeasurementAlgorithmABC):

    def __init__(self, n_bottles, n_detectors):
        self.n_bottles = n_bottles
        self.n_detectors = n_detectors
        self.remaining = {(bottle, detector) for bottle in range(n_bottles) for detector in range(n_detectors)}
        self.arr = make_empty(n_bottles, n_detectors)
        self.col_zeros = np.asarray([self.n_bottles/2]*self.n_detectors, dtype=int) # TODO move this to cellar

    def next(self):
        '''returns bottle, detector for the next measurement'''
        return self.remaining.pop() # or maybe use np.random.choice for this

    def update(self, bottle:int, detector:int, measurement:int):
        '''doesnt return anything, just gives the last measurement to the algorithm'''
        self.arr[detector][bottle] = measurement
        self._update_remaining()

    def _update_remaining(self):
        # remaining will be set of tuples of null values.
        self.remaining = set(zip(*np.where(self.arr.isnull()))) # drops nulls
        print(self.arr)
        print('n remaining', len(self.remaining))

    def done(self):
        '''returns True when algortithm has solved everything'''
        if self.arr.isnull().any().any():
            return False
        return True

class RandomWithConstraintSolving(Random):

    def update(self, bottle:int, detector:int, measurement:int):
        '''doesnt return anything, just gives the last measurement to the algorithm'''
        self.arr[detector][bottle] = measurement
        self.arr = fill_from_constraints(self.col_zeros, self.arr)
        self._update_remaining()


class EntropyComputer(RandomWithConstraintSolving):
    ''' has modes of max expected info gain, and max min info gain'''
    def next(self):
        # store the two possiblieies for the chosen state, because one of them will be correct
        # calculates entropy for each of the two possible measurement outcomes for each unknown
        ...

class EntropyWithHeuristcs:
    pass



