"""
Make a tiny debuggable version of train_x_lpd_5_phr.npz
"""

import numpy as np
import sys

if __name__ == '__main__':
    with np.load('train_x_lpd_5_phr.npz') as f:
        data = np.zeros(f['shape'], np.bool_)
        data[[x for x in f['nonzero']]] = True
    data = data[:10000]
    np.savez_compressed('train_x_lpd_5_phr_debug', data)
