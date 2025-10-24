import math

import numpy
import numpy as np
import pandas as pd
from abc import ABC
from permutation import Permutation
import math
import itertools


def make_empty(n_rows, n_cols):
    return pd.DataFrame(np.full((n_rows, n_cols), pd.NA, dtype='object'), dtype='boolean')


def make_full(n_rows, n_cols, order=0):
    df = pd.DataFrame([[i // (2 ** n) % 2 for i in range(n_rows)] for n in range(n_cols)]).T
    if order:
        n = math.factorial(n_rows)
        print(n, 'possible permutations')
        # get seed in range
        order = order % n
        # use the order as the Lehmer code to get a unique permutation
        row_order = Permutation.from_lehmer(order, n_rows).to_image(n_rows)
        # make it 0 indexed
        row_order = np.subtract(row_order, 1)
        df = df.iloc[row_order].reset_index(drop=True)
    return df


def permutation_generator(n_rows, n_cols):
    # faster for generating all permutations
    df = pd.DataFrame([[i // (2 ** n) % 2 for i in range(n_rows)] for n in range(n_cols)]).T
    for row_order in itertools.permutations(range(n_rows)):
        return df.iloc[row_order].reset_index(drop=True)


def fill_from_constraints(col_zeros, matrix):
    new_matrix = matrix.copy()
    # count each col with missing values
    cols = (matrix == 0).sum() == col_zeros
    # if a column has the maximum number of zeros, the rest must be ones
    if cols.any():
        new_matrix[cols.index[cols]] = matrix[cols.index[cols]].fillna(True)
    # vice versa
    cols = (new_matrix == 1).sum() == np.subtract(new_matrix.shape[0], col_zeros)
    if cols.any():
        new_matrix[cols.index[cols]] = new_matrix[cols.index[cols]].fillna(False)
    # if a row has n-1 entries and matches a row that is full, infer the last bit

    # good difficult interview question
    rows = matrix.isnull().sum(axis=1)
    # get all the rows that are full that can be used to infer the missing bit
    zero_missing = matrix[rows == 0]
    for i, row in matrix[rows == 1].iterrows():
        # it's easy and partially vectorized to just check matches as opposed to using a trie or something
        # is the mising bit a one?
        if (zero_missing == row.fillna(1)).all(1).any():
            new_matrix.loc[row.name].fillna(0, inplace=True)
        # is the mising bit a one?
        elif (zero_missing == row.fillna(0)).all(1).any():
            new_matrix.loc[row.name].fillna(1, inplace=True)

    # if there is a change, run again
    if (new_matrix != matrix).all().all():
        return fill_from_constraints(col_zeros, new_matrix)
    return new_matrix


class MeasurementAlgorithmABC(ABC):

    def __init__(self, n_bottles, n_detectors):
        ...

    def next(self):
        '''returns bottle, detector for the next measurement'''
        ...

    def update(self, bottle: int, detector: int, measurement: int):
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
        self.col_zeros = np.asarray([self.n_bottles / 2] * self.n_detectors, dtype=int)  # TODO move this to cellar

    def next(self):
        '''returns bottle, detector for the next measurement'''
        return self.remaining.pop()  # or maybe use np.random.choice for this

    def update(self, bottle: int, detector: int, measurement: int):
        '''doesnt return anything, just gives the last measurement to the algorithm'''
        self.arr[detector][bottle] = measurement
        self._update_remaining()

    def _update_remaining(self):
        # remaining will be set of tuples of null values.
        self.remaining = set(zip(*np.where(self.arr.isnull())))  # drops nulls
        print(self.arr)
        print('n remaining', len(self.remaining))

    def done(self):
        '''returns True when algortithm has solved everything'''
        if self.arr.isnull().any().any():
            return False
        return True


class RandomWithConstraintSolving(Random):

    def update(self, bottle: int, detector: int, measurement: int):
        '''doesnt return anything, just gives the last measurement to the algorithm'''
        self.arr[detector][bottle] = measurement
        self.arr = fill_from_constraints(self.col_zeros, self.arr)
        self._update_remaining()


def init_probability_matrix(arr):
    p = pd.DataFrame(index=arr.index, columns=arr.index)
    for i, row in arr.iterrows():
        binary = str(bin(i))[2:]
        for b, j in zip(binary, row):
            # if bottle is incompatible with measurement, set probability to 0
            if pd.notna(j) and int(b) != j:
                p[i][j] = 0
    return p


def solve_probability_matrix(p):
    '''p is our doubly stochastic matrix'''

    # aggregate to make block matrix
    row_agg = p.groupby(list(p.columns), dropna=False).apply(lambda x: frozenset(x.index))
    unique_rows = pd.DataFrame(row_agg.index.to_list()).T
    col_agg = unique_rows.groupby(list(unique_rows.columns), dropna=False).apply(lambda x: frozenset(x.index))
    eqn = pd.DataFrame(col_agg.index.to_list(), columns=row_agg.iloc[col_agg.index.names].values,
                       index=col_agg.values).T

    # make cols out of unknowns, where the column labels represent the original row and col indices
    eqn_stacked = eqn.stack(dropna=False)
    unknown_indices = eqn_stacked.index[eqn_stacked.isnull()]
    equation_indices = eqn.index.tolist() + eqn.columns.tolist()
    # make matrix A and vector b to form linear equation system.
    a = pd.DataFrame([[ai in ac for ac in unknown_indices] for ai in equation_indices], columns=unknown_indices,
                     index=equation_indices, dtype=int)
    # because we aggregated the rows, and they originally sum to one, we need to count the number of rows that were added
    b = a.index.map(len)

    # solve ax = b
    x = solve_overspecified(a, b)
    # fill unknowns back in
    for (rows, cols), value in zip(unknown_indices, x):
        # reverse operations by dividing by number of rows and cols that were combined in the groupby operation
        normalized_value = value / (len(rows) * len(cols))
        for r in rows:
            for c in cols:
                p[c][r] = normalized_value

    # make sure rows and cols sum to 1
    assert (p.sum() == 1).all()
    assert (p.sum(1) == 1).all()
    return p


from sympy import Matrix, linsolve


def solve_overspecified(A, b):
    # convert to sympy matrices
    A = Matrix(A)
    b = Matrix(b)
    sol = linsolve((A, b))
    print("Solution:", sol)
    if len(sol) != 1:
        raise ValueError(f'not one solution {sol}')
    return sol.args[0]


class GreedMaxExpEntropy(RandomWithConstraintSolving):
    ''' has modes of max expected info gain, and max min info gain'''

    def __init__(self, n_bottles, n_detectors):
        super().__init__(n_bottles, n_detectors)
        # initialize probability matrix to have uniform priors
        self.probability_matrix = init_probability_matrix(self.arr)

    def next(self):
        high_entropy_measurements = pd.DataFrame(np.nan, index=self.arr.index, columns=self.arr.columns)
        low_entropy_measurements = pd.DataFrame(np.nan, index=self.arr.index, columns=self.arr.columns)
        # calculates entropy for each of the two possible measurement outcomes for each unknown
        for bottle, detector in self.remaining:
            # calculate hypothetical measurement result
            entropies = []
            for measurement_result in 0, 1:
                temp_arr = self.arr.copy()
                temp_arr.loc[bottle, detector] = measurement_result
                # calculate probabilities
                temp_p = solve_probability_matrix(temp_arr)
                # calculate entropy
                entropies.append(compute_matrix_entropy(temp_p))

            # update high_entropy_measurements
            high_entropy_measurements.loc[bottle, detector] = max(entropies)
            # update low_entropy_measurements
            low_entropy_measurements.loc[bottle, detector] = min(entropies)
            # TODO flip the methods that use this because we're not going to calculat delta e, and low entropy is better than high entropy
        return self._decision(low_entropy_measurements, high_entropy_measurements)

    def _decision(self, low_info_gain, high_info_gain):

        return (low_info_gain + high_info_gain).idxmax()


class GreedyMaxMinEntropy(GreedMaxExpEntropy):
    def _decision(self, low_info_gain, high_info_gain):
        # return the location of the highest low info gain
        return low_info_gain.idxmax()


class GreedyMaxEntropy(GreedMaxExpEntropy):
    def _decision(self, low_info_gain, high_info_gain):
        # return the location of the highest low info gain
        return high_info_gain.idxmax()


def entropy(m):
    return - m * np.log2(m)


def compute_matrix_entropy(m: pd.DataFrame):
    return entropy(m.astype(float)).sum().sum()
