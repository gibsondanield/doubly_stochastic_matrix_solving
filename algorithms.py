import math
from inequality_solving import *
import numpy
import numpy as np
import pandas as pd
from abc import ABC
from permutation import Permutation
import math
import itertools
from sympy import Matrix, linsolve
import sympy


def decimal_to_binary_arr(n, n_bits):
    arr = np.zeros(n_bits, dtype=bool)
    for i, v in enumerate(reversed(bin(n)[2:])):
        arr[-i - 1] = v == '1'
    return arr

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
    df_original = pd.DataFrame([[i // (2 ** n) % 2 for i in range(n_rows)] for n in range(n_cols)]).T # change this to make full
    for row_order in itertools.permutations(range(n_rows)):
        df = df_original.loc[list(row_order)]
        index = df.index[:]
        yield df.reset_index(drop=True), index

def brute_force_probabilities(arr):
    i = 0
    p = pd.DataFrame(0, index=arr.index, columns=arr.index, dtype=int)
    mask_non_null = arr.notna()
    rows, cols = np.where(arr.notna())
    for df, index in permutation_generator(arr.shape[0], arr.shape[1]):
        # if i % 10000 == 0:
        #     print(i)

        has_mismatch = False
        for r, c in zip(rows, cols):
            if df.loc[r, c] != arr.loc[r, c]:
                has_mismatch = True
                break
        if not has_mismatch:
            i += 1
            for row, col in zip(p.index,index):
                p.loc[row, col] += 1
    return p.astype(object).applymap(sympy.core.numbers.Rational)/i

def fill_from_constraints(col_zeros, matrix):
    '''human-usable logic: infers missing values from column constraints'''
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
    zero_missing = matrix[rows == False]
    for i, row in matrix[rows == True].iterrows():
        # it's easy and partially vectorized to just check matches as opposed to using a trie or something
        # is the mising bit a one?
        if (zero_missing == row.fillna(True)).all(1).any():
            new_matrix.loc[row.name].fillna(False, inplace=True)
        # is the mising bit a one?
        elif (zero_missing == row.fillna(False)).all(1).any():
            new_matrix.loc[row.name].fillna(True, inplace=True)

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


def binary_to_probability_matrix(arr):
    '''todo test this round trip'''
    binary = make_full(*arr.shape)
    p = pd.DataFrame(index=arr.index, columns=arr.index)
    for i in p.index:
        for col in p.columns:
            for value, bit in zip(arr.loc[i], binary.loc[col]):
                if pd.notna(value) and value != bit:
                    p[col][i] = 0
    return p

def append_equalities(eqn, a, b):
    for i, row in eqn.iterrows():
        # if it's all nulls, skip it
        if row.isnull().all():
            continue
        null_cols = row.isnull().index[row.isnull()]
        for idx in range(len(null_cols) - 1):
            # set the adjacent values equal
            new_row_idx = len(a) + 1
            a.loc[new_row_idx] = 0
            a.loc[new_row_idx, (i, null_cols[
                idx])] = 1  # len(null_cols[idx]) # TODO make these the number of entries i think
            a.loc[new_row_idx, (i, null_cols[idx + 1])] = -1  # len(null_cols[idx+1])
            b = list(b) + [0]  # len(i) * (len(null_cols[idx]) + len(null_cols[idx+1]))]
    return a, b

def merge_cols_in_linear_system(a, col1, col2):
    ...

def make_linear_constraints_from_cols(arr, col_sums):
    # the remaining values have to add up
    ...

def find_automorphisms(a):
    import networkx as nx
    from networkx.algorithms.isomorphism import GraphMatcher
    G = nx.from_numpy_array(np.dot(a.T, a))
    matcher = GraphMatcher(G, G)

    automorphisms = []
    for mapping in matcher.isomorphisms_iter():
        new = {}
        for k, v in mapping.items():
            if k!=v:
                k,v = sorted([k,v])
                new[a.columns[k]] = a.columns[v]
        if new:
            automorphisms.append(new)

    print(f"Found {len(automorphisms)} automorphisms, {automorphisms}")

    return automorphisms

def rank(a):
    rank = np.linalg.matrix_rank(a)
    print(f'rank {rank} out of {a.shape[1]}')
    return rank

def solve_probability_matrix(p):
    '''p is our doubly stochastic matrix'''

    # aggregate to make block matrix
    row_agg = p.groupby(list(p.columns), dropna=False).apply(lambda x: frozenset(x.index))
    unique_rows = pd.DataFrame(row_agg.index.to_list()).T
    col_agg = unique_rows.groupby(list(unique_rows.columns), dropna=False).apply(lambda x: frozenset(x.index))
    eqn = pd.DataFrame(col_agg.index.to_list(), columns=row_agg.iloc[col_agg.index.names].values,
                       index=col_agg.values).T
    # eqn = p.copy()
    # make cols out of unknowns, where the column labels represent the original row and col indices
    eqn_stacked = eqn.stack(dropna=False)
    unknown_indices = eqn_stacked.index[eqn_stacked.isnull()]
    equation_indices = eqn.index.tolist() + eqn.columns.tolist()
    # make matrix A and vector b to form linear equation system.

    a = pd.DataFrame([[ai == ac[0] for ac in unknown_indices] for ai in eqn.index] +
                     [[ai == ac[1] for ac in unknown_indices] for ai in eqn.columns],
                     columns=unknown_indices,
                     index=equation_indices, dtype=int)

    original_rank = rank(a)
    print(f'original rank {original_rank}')
    # Optional: drop duplicate rows and columns
    # a = a.drop_duplicates().T.drop_duplicates().T

    # because we aggregated the rows, and they originally sum to one, we need to count the number of rows that were added
    b = a.index.map(len)
    x0 = solve_linear_system(a, b)
    # b = pd.Series(1, index = a.index)

    # try solving with inequalities
    # upper bound is the min b value that xi is involved with
    upper = []
    for col in a:
        upper.append(int(b.where(a[col].values).min()))
    try:
        *_, x_sol = solve_vector_bounds(x0, upper)
    except:
        print('no solution with inequalities')
        x_sol = None

    if x_sol is not None:
        return fill_probability_matrix(p, x_sol, unknown_indices)
    if original_rank < a.shape[1]:
        print('adding extra constraints because of rank deficiency', )
        # in rows where measurement took place.... split evenly? because of... bayesian principal of probability redistribution? like that thing ive been wondering about for a long time. Whn something disappears...everthing else is proportioniate
        # the bottle could be any of the hypothetical ones evenly, but there's no reason for the hypothetical ones to be evenly distributed? except in the abcense of other information (ie not having a row or col zero
        automorphisms = find_automorphisms(a)
        # do this recursively by deleting columns and adding rows
        for mapping in automorphisms:
            # add equality constraint
            for k,v in mapping.items():
                # set the adjacent values equal
                new_row_idx = len(a) + 1
                a.loc[new_row_idx] = 0
                a.loc[new_row_idx, k] = 1  # len(null_cols[idx]) # TODO make these the number of entries i think
                a.loc[new_row_idx, v] = -1  # len(null_cols[idx+1])
                b = list(b) + [0]  # len(i) * (len(null_cols[idx]) + len(null_cols[idx+1]))]
            new_rank = np.linalg.matrix_rank(a)
            print('new rank', new_rank)
            if new_rank>=a.shape[1]:
                break
        if len(automorphisms) == 0:
            print('unknown rank deficiency NO AUTOMORPHISMS')
        # # does eqn have unknown col?
        # if (eqn.isnull().mean()==1).any():
        #     print('full null col in eqn')
        #     # set the other vars equal
        #     a, b = append_equalities(eqn, a, b)
        # if (eqn.isnull().mean(1) == 1).any():
        #     print('full null col in eqn')
        #     # set the other vars equal
        #     # a, b = append_equalities(eqn.T, a, b)
        # if np.linalg.matrix_rank(a) < a.shape[1]:

            # raise ValueError('unknown rank deficiency')
        # for i, (j,k) in enumerate(((7,6),(1,0))):
        # for i, (j,k) in enumerate(((0,6),(1,7),(3,4),(1,0))):
        #     a.loc[i] = 0
        #     a.loc[i,a.columns[j]] = 1
        #     a.loc[i,a.columns[k]] = -1
        #     b = list(b) + [0]

    new_rank = rank(a)
    print(f'rank {original_rank} -> {new_rank} out of {a.shape[1]}')
    # solve ax = b
    x = solve_linear_system(a, b)
    if x.free_symbols:
        print(list(zip(x0, x)))
        print(pd.DataFrame(list(zip(*sympy.linear_eq_to_matrix(x, tuple(x.free_symbols)),b))).drop_duplicates())
        *_, x = solve_vector_bounds(x, upper)

    return fill_probability_matrix(p, x, unknown_indices)

def fill_probability_matrix(p, x, unknown_indices):
    # fill unknowns back in
    p_filled = pd.DataFrame(index=p.index, columns = p.columns, dtype = object)
    for (rows, cols), value in zip(unknown_indices, x):
        # reverse operations by dividing by number of rows and cols that were combined in the groupby operation
        # make sure this is sp.core.numbers.Rational
        normalized_value = value / (len(rows) * len(cols))
        for r in rows:
            for c in cols:
                p_filled.loc[r, c] = normalized_value
    # null values will be the zeros we had before
    p_filled = p_filled.fillna(sympy.core.numbers.Rational(0))
    # make sure rows and cols sum to 1
    if not ((p_filled.sum() - 1).abs() < 10E-6).all():
        raise ValueError(f'cols dont sum to 1, {p_filled.sum()}')
    if not ((p_filled.sum(1) - 1).abs() < 10E-6).all():
        raise ValueError(f'rows dont sum to 1, {p_filled.sum(1)}')
    if (p_filled<0).any().any():
        raise ValueError(f'negative values in p_filled')
    return p_filled


def probability_matrix_to_binary_probabilties(p,n_bits):
    '''gives the probability of each bit being 1
    could vectorize this'''
    binary_probabilities = pd.DataFrame(0, index=p.index, columns=range(n_bits), dtype=float)
    for col in p.columns:
        binary_vector = (decimal_to_binary_arr(col, n_bits))
        for i, probability in p[col].items():
            binary_probabilities.loc[i] += probability * binary_vector
    return binary_probabilities


def solve_linear_system(A, b):
    # convert to sympy matrices
    A = Matrix(A)
    b = Matrix(b)
    sol = linsolve((A, b))
    print("Solution:", sol)
    if len(sol) != 1:
        raise ValueError(f'not one solution {sol}')
    return sol.args[0]


class GreedyMaxExpEntropy(Random):
    '''Maximizes the expected information gain of the measurement'''

    def __init__(self, n_bottles, n_detectors):
        super().__init__(n_bottles, n_detectors)
        # initialize probability matrix to have uniform priors
        self.probability_matrix = binary_to_probability_matrix(self.arr)

    def next(self):
        high_entropy_measurements = pd.DataFrame(np.nan, index=self.arr.index, columns=self.arr.columns)
        low_entropy_measurements = pd.DataFrame(np.nan, index=self.arr.index, columns=self.arr.columns)
        # calculates entropy for each of the two possible measurement outcomes for each unknown
        for bottle, detector in self.remaining:
            # calculate hypothetical measurement result
            entropies = []
            for measurement_result in False, True:
                temp_arr = self.arr.copy()
                temp_arr.loc[bottle, detector] = measurement_result
                # calculate probabilities
                temp_p = solve_probability_matrix(binary_to_probability_matrix(temp_arr))
                # calculate entropy
                entropies.append(compute_matrix_entropy(temp_p))

            # update high_entropy_measurements
            high_entropy_measurements.loc[bottle, detector] = max(entropies)
            # update low_entropy_measurements
            low_entropy_measurements.loc[bottle, detector] = min(entropies)
            # TODO flip the methods that use this because we're not going to calculat delta e, and low entropy is better than high entropy
        return self._decision(low_entropy_measurements, high_entropy_measurements)

    def _decision(self, low_info_gain, high_info_gain):

        return (low_info_gain + high_info_gain).stack().idxmax()

    def update(self, bottle: int, detector: int, measurement: int):
        self.arr[detector][bottle] = measurement
        # see if we've solved anything
        self.probability_matrix = binary_to_probability_matrix(self.arr)
        binary_probabilities = probability_matrix_to_binary_probabilties(self.probability_matrix, self.n_detectors)
        # if this solved anything outright, update arr with the zeros and ones
        self.arr[binary_probabilities==0]=0 # watch out for floating point errors
        self.arr[binary_probabilities==1]=1
        # log entropy values
        print('remaining entropy', compute_matrix_entropy(self.probability_matrix))
        self._update_remaining()

class GreedyMaxMinEntropy(GreedyMaxExpEntropy):
    def _decision(self, low_info_gain, high_info_gain):
        # return the location of the highest low info gain
        return low_info_gain.idxmax()


class GreedyMaxEntropy(GreedyMaxExpEntropy):
    '''What decision gives us the highest minimum guaranteed information gain?'''
    def _decision(self, low_info_gain, high_info_gain):
        # return the location of the highest low info gain
        return high_info_gain.idxmax()


def entropy(p):
    '''computes shannon entropy of probability p '''
    return - p * np.log2(p)


def compute_matrix_entropy(m: pd.DataFrame):
    '''computes total shannon entropy of probability matrix m '''
    return entropy(m.astype(float)).sum().sum()
