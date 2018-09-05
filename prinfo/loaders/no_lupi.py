import os

import numpy as np

class ARDSSubsampledEHRLoader:

    def __init__(self, 
        csv_path,
        uncertain=False,
        lazy=True):

        self.csv_path = csv_path
        self.uncertain = uncertain
        self.lazy = lazy

        self.data = None

        if not self.lazy:
            self._set_data()

    def get_data(self):

        if self.data is None:
            self._set_data()

        return self.data

    def _set_data(self):

        with open(self.csv_path) as f:
            lines = [l.strip().split(',')
                     for l in f]
            as_numbers = [[float(i) for i in l]
                          for l in lines]
            as_np_array = np.array(as_numbers)
            X = np.hstack([
                as_np_array[:,8:-1], 
                np.ones((as_np_array.shape[0], 1))])
            y = as_np_array[:,2][:,np.newaxis]
            c = as_np_array[:,3][:,np.newaxis]

            if self.uncertain:
                self.data = (X, y, c)
            else:
                self.data = (X, y)

    def cols(self):

        return self.data[0].shape[1]

    def rows(self):

        rows = 0

        if self.data is not None:
            rows = self.data[0].shape[0]

        return rows

    def refresh(self):

        self.data = None
