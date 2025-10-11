import numpy as np
import pandas as pd
from abc import ABC


def make_empty(n_rows,n_cols):
    return pd.DataFrame(np.full((n_rows, n_cols), pd.NA, dtype='object'), dtype='boolean')

def make_full(n_rows,n_cols):
    df = pd.DataFrame([[i // (2**n) % 2 for i in range(n_rows)] for n in range(n_cols)]).T
    return df[df.columns[::-1]]

def fill_from_constraints(col_zeros, matrix):

    # count each col with missing values
    cols = (matrix==0).sum() == col_zeros
    # if a column has the maximum number of zeros, the rest must be ones
    new_matrix = matrix[cols.index[cols]].fillna(1)
    # vice versa
    cols = (new_matrix == 1).sum() == new_matrix.shape[0] - col_zeros
    new_matrix = new_matrix[cols.index[cols]].fillna(0)
    # if a row has n-1 entries and matches a row that is full, infer the last bit
    rows = matrix.isnull().sum(axis=1) == 1

    # if there is a change, run again
    if (new_matrix!=matrix).all().all():
        return fill_from_constraints(col_zeros, new_matrix)
    return matrix



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

class RandomWithConstraintSolving(Random):

    def update(self):
        '''uses row and col constraint satisfaction'''

class EntropyComputer(MeasurementAlgorithmABC):
    ''' has modes of max expected info gain, and max min info gain'''
    def next(self):
        # store the two possiblieies for the chosen state, because one of them will be correct
        ...
    def update(self):
        ...

