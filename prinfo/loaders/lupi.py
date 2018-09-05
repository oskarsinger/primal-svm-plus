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

class ARDSSubsampledEHRLUPIComparisonLoader:

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
            y_binary = np.zeros_like(y)
            y_binary[y == 1] = 1

            # Turn x-ray-based-rating into binary representation
            pre_X_p = as_np_array[:,-1]
            reduced_pre_X_p = np.zeros((pre_X_p.shape[0], 2))
            reduced_pre_X_p[pre_X_p > 0,0] = 1
            reduced_pre_X_p[pre_X_p > 4,1] = 1

            # Compute privileged info based on compatibility with binary y
            expanded_pre_X_p = np.zeros((pre_X_p.shape[0], 5))
            pos_and_agree = np.logical_and(
                reduced_pre_X_p[:,1] == 1,
                reduced_pre_X_p[:,1] == y[:,0])
            expanded_pre_X_p[:,0] = reduced_pre_X_p[:,0]
            expanded_pre_X_p[pos_and_agree,1] = 1
            neg_and_agree = np.logical_and(
                reduced_pre_X_p[:,1] == 0,
                reduced_pre_X_p[:,1] == y[:,0])
            expanded_pre_X_p[neg_and_agree,2] = 1
            pos_and_disagree = np.logical_and(
                reduced_pre_X_p[:,1] == 1,
                np.logical_not(reduced_pre_X_p[:,1] == y[:,0]))
            expanded_pre_X_p[pos_and_disagree,3] = 1
            neg_and_disagree = np.logical_and(
                reduced_pre_X_p[:,1] == 0,
                np.logical_not(reduced_pre_X_p[:,1] == y[:,0]))
            expanded_pre_X_p[neg_and_disagree,4] = 1
            X_p = np.hstack([
                expanded_pre_X_p,
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

class ARDSSubsampledEHRMissingLUPILoader:

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
            pre_X_p = as_np_array[:,-1]
            X_o = np.hstack([
                as_np_array[:,8:-1], 
                np.ones((as_np_array.shape[0], 1))])
            X_o_missing = X_o[pre_X_p == 0,:]
            X_o_not_missing = X_o[pre_X_p > 0,:]
            y_missing = as_np_array[:,2][pre_X_p == 0,np.newaxis]
            y_not_missing = as_np_array[:,2][pre_X_p > 0,np.newaxis]

            X_p = np.hstack([
                np.zeros((X_o_not_missing.shape[0], 1)),
                np.ones((X_o_not_missing.shape[0], 1))])
            pre_X_p = pre_X_p[pre_X_p > 0]
            X_p[pre_X_p > 4,0] = 1


            if self.uncertain:
                c = as_np_array[:,3][:,np.newaxis]
                self.data = (X_o_missing, X_o_not_missing, X_p, y_missing, y_not_missing, c)
            else:
                self.data = (X_o_missing, X_o_not_missing, X_p, y_missing, y_not_missing)

    def cols(self):

        cols = 0

        if self.data is not None:
            c_o = self.data[0].shape[1]
            c_p = self.data[2].shape[1]
            cols = c_o + c_p

        return cols

    def rows(self):

        rows = 0

        if self.data is not None:
            rows_missing = self.data[0].shape[0]
            rows_not_missing = self.data[1].shape[0]
            rows = rows_missing + rows_not_missing

        return rows

    def refresh(self):

        self.data = None

