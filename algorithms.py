import numpy as np
import pandas as pd
from abc import ABC

from matrix_utilities import make_empty, fill_from_constraints, binary_to_probability_matrix, solve_probability_matrix, \
    probability_matrix_to_binary_probabilties, compute_matrix_entropy


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


