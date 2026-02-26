import numpy as np

class MahalanobisWeighter:
    """
    Converts a Mahalanobis distance into a weighting factor w.

    Parameters
    ----------
    wmin : float
        Minimum weight value (for large distances).
    wmax : float
        Maximum weight value (for small distances).
    mth : float
        Distance threshold for clipping.
    """

    def __init__(self, wmin, wmax, mth):
        self.wmin = wmin
        self.wmax = wmax
        self.mth = mth

    def compute(self, dist):
        """
        Compute the weight for a given distance or array of distances.
        """
        tc = np.clip(dist, 0, self.mth) / self.mth
        w = self.wmin + (1 - tc) * (self.wmax - self.wmin)
        return w