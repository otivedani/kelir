import numpy as np

"""
    Empirical Cumulative Distribution Function
"""


def ecdf(input_array, l_pad=None, r_pad=None):
    if len(input_array.shape) < 2:
        raise ValueError("Expecting numpy array with shape minimum of 2,\
                            to make last dimension broadcastable.")

    _l, _r = (l_pad is not None), (r_pad is not None)
    last = input_array.shape[-1]
    dsize = input_array.size // last

    # arrange new array and sort
    x = np.empty((last, dsize + _l + _r))
    x[:, 0] = l_pad
    x[:, 1] = r_pad
    x[:, _l+_r:] = input_array.reshape(dsize, last).swapaxes(1, 0)
    x.sort(axis=1)

    # ecdf of x : 0, 1/(w*h) ... (w*h)/(w*h), 1
    ecdf_x_ = np.empty((dsize + _l + _r))
    if _l:
        ecdf_x_[0] = 0.
    if _r:
        ecdf_x_[-1] = 1.
    ecdf_x_[_l:dsize+1] = np.linspace(1/dsize, 1., dsize, endpoint=True)

    return x, ecdf_x_
