import itertools
import math
import pandas as pd
from permutation import Permutation
import inequality_solving
import numpy as np
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
    rows, cols = np.where(arr.notna())
    for df, index in permutation_generator(arr.shape[0], arr.shape[1]):
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

import networkx as nx
# from networkx.algorithms.isomorphism import GraphMatcher

def recursive_merge(G):
    def merge_automorphic_nodes(G):
        auts = nx.algorithms.isomorphism.GraphMatcher(G, G)
        auts_list = list(auts.isomorphisms_iter())
        # Compute node orbits
        orbits = []
        seen = set()
        for a in auts_list:
            for v in G.nodes:
                if v not in seen:
                    orbit = {v2 for v2, v1 in a.items() if a[v2] == v}
                    if orbit:
                        orbits.append(orbit)
                        seen |= orbit
        # Merge nodes in same orbit
        merged = nx.Graph()
        for i, orbit in enumerate(orbits):
            merged.add_node(i)
        for i, orbit_i in enumerate(orbits):
            for j, orbit_j in enumerate(orbits):
                if i != j and any(G.has_edge(u, v) for u in orbit_i for v in orbit_j):
                    merged.add_edge(i, j)
        return merged

    # G = nx.path_graph(6)
    n=0
    while True:
        new_G = merge_automorphic_nodes(G)
        if len(new_G) == len(G):
            break
        G = new_G
        n+=1
    print('after iterations',n, G.nodes, G.edges)
    return G


import scipy as sp
def find_automorphisms(a, b): # TODO make simple unit test
    '''indexed by a columns'''
    import networkx as nx
    from networkx.algorithms.isomorphism import GraphMatcher
    G = nx.from_numpy_array(np.dot(a.T, a).astype(bool), )
    # use upper as proxy for similarity
    # for i, col in enumerate(a.columns):
    #     for value in b.where(a[col].astype(bool)):
    #         G.add_edge(i, str(value))
    gm = recursive_merge(G)
    matcher = GraphMatcher(G, G)
    automorphisms = []

    for mapping in matcher.isomorphisms_iter():
        new = {}
        for k, v in mapping.items():

            # exclude the self and the b values that were turned to strings
            if k!=v and type(k)==int and type(v)==int:
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

def matrix_aggregation(df):
    '''makes a block matrix, also aggregating indices'''
    row_agg = df.groupby(list(df.columns), dropna=False).apply(lambda x: frozenset(x.index))
    unique_rows = pd.DataFrame(row_agg.index.to_list()).T
    col_agg = unique_rows.groupby(list(unique_rows.columns), dropna=False).apply(lambda x: frozenset(x.index))
    eqn = pd.DataFrame(col_agg.index.to_list(), columns=row_agg.iloc[col_agg.index.names].values,
                       index=col_agg.values).T
    return eqn

def merge_cols_for_equalities(a, col1, col_list):
    '''adds values to co1 1 and deletes others'''
    a[col1] += a[col_list].sum(1)
    return a.drop(columns=col_list)

def set_col_to_value(a, b, col, value):
    '''substitutes value for col in linear system a, b'''
    b -= a[col] * value
    del a[col]
    return a, b

def get_upper_bounds(a,b):
    return [int(b.where(a[col].values).min()) for col in a]

def add_automorphic_constraints(a,b):
    automorphisms = find_automorphisms(a, b)
    b=list(b)
    rank(a)
    # make one disjoint set containing automorphic groups
    ds = sp.cluster.hierarchy.DisjointSet()
    for mapping in automorphisms:
        # add equality constraint
        for k, v in mapping.items():
            ds.add(k)
            ds.add(v)
            ds.merge(k, v)
    for subset in ds.subsets():
        ss = list(subset)
        for i in range(1,len(subset)):
            # make 2 inequality constraints to make an equality constraint, always constraining to the zeroth element
            for j, k in ((0, i),):# ,(i,0):
                # inefficiently add a row to a initialized as zeros
                idx = str((ss[j],ss[k]))
                a.loc[idx] = 0
                a.loc[idx, ss[j]] = np.prod(tuple(map(len,ss[j]))) #
                a.loc[idx, ss[k]] = -np.prod(tuple(map(len,ss[k])))
                b.append(0)
    print(a)
    rank(a)
    return a, pd.Series(b, index=a.index)


def solve_probability_matrix(p):
    '''p is our doubly stochastic matrix'''

    # aggregate to make block matrix
    eqn = matrix_aggregation(p)
    print(eqn)
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
    # because we aggregated the rows, and they originally sum to one, we need to count the number of rows that were added
    b = a.index.map(len)

    # x0 = inequality_solving.solve_linear_system(a, b)

    # upper bound is the min b value that xi is involved with
    upper = get_upper_bounds(a, b)

    res = inequality_solving.solve_lp(a, b, upper)
    # try:
    #     *_, x_sol = inequality_solving.solve_vector_bounds(x0, upper)
    # except:
    #     print('no solution with inequalities')
    #     print('linprog bounds', inequality_solving.solve_vector_bounds_linprog(x0, upper))
    #     x_sol = None
    #
    # if x_sol is not None:
    #     return fill_probability_matrix(p, x_sol, unknown_indices)
    a, b = add_automorphic_constraints(a,b)
    inequality_solving.solve_lp(a, b, upper)
    print
    while len(res) < a.shape[1]:
        # substitute any results
        for col, value in res.items():
            a, b = set_col_to_value(a, b, col, value)
        automorphisms = find_automorphisms(a, b)
        # do this recursively by deleting columns and adding rows
        ds = sp.cluster.hierarchy.DisjointSet()
        for mapping in automorphisms:
            # add equality constraint
            for k,v in mapping.items():
                ds.add(k)
                ds.add(v)
                ds.merge(k, v)
                # remove from upper as well
                upper.pop(a.columns.get_loc(v))
                a = merge_cols_for_equalities(a, k, [v])
            new_rank = np.linalg.matrix_rank(a)
            print('new rank', new_rank)
            res = inequality_solving.solve_lp(a, b, upper)
            if len(res)>=a.shape[1]:
                break
            break
        if len(automorphisms) == 0:
            print('unknown rank deficiency NO AUTOMORPHISMS')
        if len(res)==0:
            raise

    new_rank = rank(a)
    print(f'rank {original_rank} -> {new_rank} out of {a.shape[1]}')
    res = inequality_solving.solve_lp(a, b, upper)
    ...
    # solve ax = b
    # x = inequality_solving.solve_linear_system(a, b)
    # if x.free_symbols:
    #     print(list(zip(x0, x)))
    #     print(list(zip(x, upper)))
    #     print(pd.DataFrame(list(zip(*sympy.linear_eq_to_matrix(x, tuple(x.free_symbols)),upper))).drop_duplicates())
    #     print('linprog bounds', inequality_solving.solve_vector_bounds_linprog(x, upper))
    #     *_, x = inequality_solving.solve_vector_bounds(x, upper)
    #
    # return fill_probability_matrix(p, x, unknown_indices)


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


def entropy(p):
    '''computes shannon entropy of probability p. Factor of 2 included because it's always a binary outcome '''
    return - 2 * p * np.log2(p)


def compute_matrix_entropy(m: pd.DataFrame):
    '''computes total shannon entropy of probability matrix m '''
    return entropy(m.astype(float)).sum().sum()
