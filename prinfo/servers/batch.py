import numpy as np

class BatchServer:

    def __init__(self, 
        data_loader, 
        center=False,
        num_coords=None):

        self.dl = data_loader
        self.center = center

    def get_data(self):

        data = self.dl.get_data()

        if self.center:
            data -= np.mean(data, axis=0)

        return data

    def cols(self):

        cols = self.dl.cols()

        if self.num_coords is not None:
            cols = self.num_coords

        return cols

    def rows(self):

        return self.dl.rows()

    def name(self):

        return self.dl.name()

    def refresh(self):

        return None

    def get_status(self):

        return {
            'data_loader': self.dl,
            'online': False}
