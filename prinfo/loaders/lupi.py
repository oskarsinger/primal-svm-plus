import os

import numpy as np

from .utils import get_one_hots as get_oh

class ARDSSubsampledEHRLUPILoader:

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
            # Set text as numpy array
            lines = [l.strip().split(',')
                     for l in f]
            as_numbers = [[float(i) for i in l]
                          for l in lines]
            as_np_array = np.array(as_numbers)

            # Get labels and binary version for privileged info
            y = as_np_array[:,2][:,np.newaxis]

            # Turn x-ray-based-rating into binary representation
            pre_X_p = as_np_array[:,-1]
            reduced_pre_X_p = np.zeros((pre_X_p.shape[0], 2))
            reduced_pre_X_p[pre_X_p > 0,0] = 1
            reduced_pre_X_p[pre_X_p > 4,1] = 1
            X_p = np.hstack([
                reduced_pre_X_p,
                np.ones((pre_X_p.shape[0], 1))])

            # Set observable info and labels
            X_o = np.hstack([
                as_np_array[:,8:-1], 
                np.ones((as_np_array.shape[0], 1))])

            if self.uncertain:
                c = as_np_array[:,3][:,np.newaxis]
                self.data = (X_o, X_p, y, c)
            else:
                self.data = (X_o, X_p, y)

    def cols(self):

        cols = 0

        if self.data is not None:
            c_o = self.data[0].shape[1]
            c_p = self.data[1].shape[1]
            cols = c_o + c_p

        return cols

    def rows(self):

        rows = 0

        if self.data is not None:
            rows = self.data[0].shape[0]

        return rows

    def refresh(self):

        self.data = None
