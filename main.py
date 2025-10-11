import numpy as np
from algorithms import *
import pandas as pd


class Cellar:

    def __init__(self, n, seed=None):
        self.measurements=0
        self.arr = make_full(2**n, n)

        if seed is not None:
            self.arr = self.arr.sample(frac=1, random_state=seed)

    def measure(self, bottle:int, detector:int):
        self.measurements+=1
        return self.arr[bottle, detector]

def entropy(m):
    return - m * np.log2(m)

def compute_entropy(m):
    return entropy(m).sum().sum()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    detectors = 3
    cellar = Cellar(detectors)
    methods = [] # random, max exp ent, max min ent,
    bottles = 2**detectors
    for m in methods:
        algo = m(bottles, detectors)
        while not m.done():
            bottle, detector = m.next()
            measurement = cellar.measure(bottle, detector)
            m.update(bottle, detector, measurement)
        assert m.answer() == cellar.arr
        print(m, cellar.measurements)
        cellar.measurements = 0

'''
method
done
next 
updatte
'''