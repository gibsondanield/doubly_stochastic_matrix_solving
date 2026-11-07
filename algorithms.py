import numpy as np
import pandas as pd
from abc import ABC
from inequality_solving import NoSolutionError

from matrix_utilities import make_empty, fill_from_constraints, binary_to_probability_matrix, solve_probability_matrix, \
    probability_matrix_to_binary_probabilties, compute_matrix_entropy, exact_probability_matrix


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
                try:
                    temp_p = solve_probability_matrix(binary_to_probability_matrix(temp_arr))
                except NoSolutionError:
                    # this outcome is impossible given what we've already measured
                    continue
                # calculate entropy
                entropies.append(compute_matrix_entropy(temp_p))
            if len(entropies) < 2:
                # only one possible outcome -> the cell is already determined, measuring it gains nothing
                continue

            # update high_entropy_measurements
            high_entropy_measurements.loc[bottle, detector] = max(entropies)
            # update low_entropy_measurements
            low_entropy_measurements.loc[bottle, detector] = min(entropies)
        return self._decision(low_entropy_measurements, high_entropy_measurements)

    def _decision(self, low_info_gain, high_info_gain):
        # The two outcomes are summed unweighted. With outcome probability q the remaining entropy
        # after each is ~ H + log2(q) and H + log2(1-q), so the sum peaks at q = 1/2: taking the MAX
        # picks the most balanced measurement, which is the most informative one.
        return (low_info_gain + high_info_gain).stack().idxmax()

    # how the probability grid is computed; subclasses can swap in exact_probability_matrix
    solver = staticmethod(solve_probability_matrix)

    def _solve(self):
        return self.solver(binary_to_probability_matrix(self.arr)).astype(float)

    def update(self, bottle: int, detector: int, measurement: int):
        self.arr[detector][bottle] = measurement
        # see if we've solved anything
        self.probability_matrix = self._solve()
        binary_probabilities = probability_matrix_to_binary_probabilties(self.probability_matrix, self.n_detectors)
        # if this solved anything outright, update arr with the zeros and ones.
        # The LP fill is an average of extreme feasible points, so its support is every cell that can be
        # nonzero in *some* consistent assignment: a bit at probability 0/1 here is genuinely forced.
        self.arr[np.isclose(binary_probabilities, 0, atol=1e-9)] = False
        self.arr[np.isclose(binary_probabilities, 1, atol=1e-9)] = True
        # log entropy values
        print('remaining entropy', compute_matrix_entropy(self.probability_matrix))
        self._update_remaining()

class GreedyMinExpEntropy(GreedyMaxExpEntropy):
    '''Same scoring as GreedyMaxExpEntropy, but picks the LOWEST unweighted sum. That looks like
    "most information" but actually favours long shots (one very unlikely, very informative
    outcome) -- worse on average and much higher variance. Kept for comparison.'''

    def _decision(self, low_info_gain, high_info_gain):
        return (low_info_gain + high_info_gain).stack().idxmin()


class GreedyMaxMinEntropy(GreedyMaxExpEntropy):
    def _decision(self, low_info_gain, high_info_gain):
        # return the location of the highest low info gain
        return low_info_gain.idxmax()


class GreedyMaxEntropy(GreedyMaxExpEntropy):
    '''What decision gives us the highest minimum guaranteed information gain?'''
    def _decision(self, low_info_gain, high_info_gain):
        # return the location of the highest low info gain
        return high_info_gain.idxmax()




class GreedyClosestToHalf(GreedyMaxExpEntropy):
    '''If every consistent arrangement is equally likely, measuring a cell whose bit is 1 with
    probability q gains exactly H2(q) bits, which is largest at q = 1/2. So: measure the unknown bit
    closest to 50/50. One probability solve per step, instead of two per candidate cell.'''

    def __init__(self, n_bottles, n_detectors):
        super().__init__(n_bottles, n_detectors)
        self.probability_matrix = self._solve()

    def next(self):
        bits = probability_matrix_to_binary_probabilties(self.probability_matrix, self.n_detectors)
        # sorted() makes ties break deterministically
        return min(sorted(self.remaining), key=lambda bd: abs(bits.loc[bd[0], bd[1]] - 0.5))


class GreedyClosestToHalfExact(GreedyClosestToHalf):
    '''Same rule, with exact probabilities (permanent ratios) instead of the LP fill-in.'''
    solver = staticmethod(exact_probability_matrix)


class GreedyMaxExpEntropyExact(GreedyMaxExpEntropy):
    '''Original greedy scoring, with exact probabilities instead of the LP fill-in.'''
    solver = staticmethod(exact_probability_matrix)
