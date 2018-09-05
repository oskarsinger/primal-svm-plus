import numpy as np

class LinearSVMPlusLoader:

    def __init__(self,
        observable_loader,
        privileged_loader,
        fraction_flipped=0.2,
        max_distance_from_margin=5,
        bias=False,
        w_o_init=None,
        w_p_init=None):

        self.oloader = observable_loader
        self.ploader = privileged_loader
        self.fraction_flipped = fraction_flipped
        self.max_distance = max_distance_from_margin
        self.bias = bias

        self.o_cols = self.oloader.cols()
        self.p_cols = self.ploader.cols()
        self.n = self.oloader.rows()

        if w_o_init is None:
            w_o_init = np.random.randn(self.o_cols, 1)

        self.w_o = w_o_init

        if w_p_init is None:
            w_p_init = np.random.randn(self.p_cols, 1)

        self.w_p = w_p_init
        self.data = None

    def get_data(self):

        if self.data is None:
            self._set_data()

        return self.data

    def _set_data(self):

        unnormed_X_o = self.oloader.get_data()
        unnormed_Xw_o = np.dot(unnormed_X_o, self.w_o)
        unnormed_X_p = self.ploader.get_data()
        unnormed_Xw_p = np.dot(unnormed_X_p, self.w_p)
        X_p = unnormed_X_p * np.sign(unnormed_Xw_p)
        Xw_p = unnormed_Xw_p * np.sign(unnormed_Xw_p)
        diff = (1 - Xw_p)[:,0]
        X_o = np.zeros_like(unnormed_X_o)
        y = np.zeros((self.n, 1))

        # Compute e, y, X_o for positive diff
        e_pos = diff[diff > 0][:,np.newaxis] * unnormed_Xw_o[diff > 0,:]
        y[diff > 0,:] = np.sign(e_pos)
        X_o[diff > 0,:] = np.power(np.absolute(e_pos), -1) * \
            unnormed_X_o[diff > 0,:]
        X_o[diff > 0,:] *= np.random.uniform(
            low=1, 
            high=self.max_distance,
            size=e_pos.shape[0])[:,np.newaxis]
        
        # Compute e, y, X_o for negative diff
        e_neg = diff[diff < 0][:,np.newaxis] * unnormed_Xw_o[diff < 0,:]
        y[diff < 0] = np.sign(e_neg)
        X_o[diff < 0,:] = np.power(np.absolute(e_neg), -1) * \
            unnormed_X_o[diff < 0,:]
        X_o[diff < 0] *= - np.random.uniform(
            low=1, 
            high=self.max_distance,
            size=e_neg.shape[0])[:,np.newaxis]

        # Compute y, X_o for zero diff
        y[diff == 0] = np.sign(unnormed_Xw_o[diff == 0,:])
        X_o[diff == 0,:] = unnormed_X_o[diff == 0,:] * \
            np.power(np.absolute(unnormed_Xw_o[diff == 0,:]), -1)
        X_o[diff == 0,:] *= np.random.uniform(
            low=1, 
            high=self.max_distance,
            size=diff[diff == 0].shape[0])[:,np.newaxis]

        if self.fraction_flipped is not None:
            size = int(self.fraction_flipped * self.oloader.rows())
            flipped = np.random.choice(
                self.oloader.rows(),
                size,
                replace=False)
            y[flipped] = -y[flipped]

        self.data = (X_o, X_p, y)

    def name(self):

        return 'SVMPlusLoader'

    def cols(self):

        return self.r_cols + self.p_cols

    def rows(self):

        return self.n
