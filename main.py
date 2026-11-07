"""Compare measurement strategies on shuffled wine cellars.

    uv run python main.py                 # 3 detectors (8 bottles), 20 cellars
    uv run python main.py 3 200 --fast    # skip the slow greedies
"""
import argparse
import statistics

from algorithms import (Random, RandomWithConstraintSolving, GreedyMaxExpEntropy, GreedyMinExpEntropy,
                        GreedyClosestToHalf, GreedyClosestToHalfExact)
from matrix_utilities import make_full


class Cellar:
    '''The hidden truth: 2^n bottles, each with a distinct n-bit code, optionally shuffled.'''

    def __init__(self, n, seed=None):
        self.measurements = 0
        self.arr = make_full(2 ** n, n)

        if seed is not None:
            self.arr = self.arr.sample(frac=1, random_state=seed).reset_index(drop=True)
        self.arr = self.arr.astype(bool)

    def measure(self, bottle: int, detector: int):
        self.measurements += 1
        return self.arr.iloc[bottle, detector]

    def reset(self):
        self.measurements = 0


def run(methods, detectors, bottles, seed=None):
    '''Runs each strategy against the same cellar; returns {strategy name: measurements used}.'''
    cellar = Cellar(detectors, seed=seed)
    measurements = {}
    for M in methods:
        m = M(bottles, detectors)
        while not m.done():
            bottle, detector = m.next()
            m.update(bottle, detector, cellar.measure(bottle, detector))
        assert (m.arr == cellar.arr).all().all(), f'{M.__name__} reconstructed the cellar incorrectly'
        measurements[M.__name__] = cellar.measurements
        cellar.reset()
    return measurements


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('detectors', type=int, nargs='?', default=3)
    parser.add_argument('cellars', type=int, nargs='?', default=20)
    parser.add_argument('--fast', action='store_true', help='skip the two-solves-per-cell greedies')
    args = parser.parse_args()

    methods = [Random, RandomWithConstraintSolving, GreedyClosestToHalf, GreedyClosestToHalfExact]
    if not args.fast:
        methods += [GreedyMaxExpEntropy, GreedyMinExpEntropy]

    results = {M.__name__: [] for M in methods}
    for seed in range(args.cellars):
        for name, n in run(methods, args.detectors, 2 ** args.detectors, seed=seed).items():
            results[name].append(n)
        print(f'cellar {seed + 1}/{args.cellars} done', flush=True)

    print(f'\nmeasurements over {args.cellars} cellars ({2 ** args.detectors} bottles):')
    for name, v in results.items():
        print(f'  {name:28s} mean {statistics.mean(v):6.2f}  min {min(v):3d}  max {max(v):3d}')
