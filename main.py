import numpy as np
from algorithms import *
import pandas as pd


class Cellar:

    def __init__(self, n, seed=None):
        self.measurements=0
        self.arr = make_full(2**n, n)

        if seed is not None:
            self.arr = self.arr.sample(frac=1, random_state=seed)
        self.arr = self.arr.astype(bool)

    def measure(self, bottle:int, detector:int):
        self.measurements+=1
        return self.arr.iloc[bottle, detector]

    def reset(self):
        self.measurements=0

def entropy(m):
    return - m * np.log2(m)

def compute_entropy(m):
    return entropy(m).sum().sum()

def run(methods, detectors, bottles):
    cellar = Cellar(detectors)
    measurements = {}
    for M in methods:
        m = M(bottles, detectors)
        while not m.done():
            bottle, detector = m.next()
            measurement = cellar.measure(bottle, detector)
            m.update(bottle, detector, measurement)
        assert (m.arr == cellar.arr).all().all()
        print(m, 'completed successfully in', cellar.measurements)
        measurements[m] = cellar.measurements
        # reset the cellar measurements
        cellar.reset()
    return measurements

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    detectors = 3

    bottles = 2**detectors
    methods = [
                Random,
               RandomWithConstraintSolving] # random, max exp ent, max min ent,
    run(methods, detectors, bottles)

'''
method
done
next 
updatte
'''