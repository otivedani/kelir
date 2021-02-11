import numpy as np
from typing import Union, Tuple


def ecdf(in_vectors: np.ndarray,
         l_pad: Union[int, None] = None,
         r_pad: Union[int, None] = None,
         reduce_to: Union[int, None] = None,
         return_cumprob: bool = False,
         ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Empirical Cumulative Distribution Function"""
    if len(in_vectors.shape) < 2:
        raise ValueError('Expected >2D array,\
                            broadcastable to last dimension')

    return_reduced = isinstance(reduce_to, int)
    if return_reduced:
        if reduce_to > in_vectors[..., 0].size:
            raise ValueError('cannot reduce to higher value than {}'.format(in_vectors[..., 0].size))
        # if (reduce_to % in_vectors[..., 0].size) != 0
        #     raise ValueError('')

    # shaping new vector
    _l, _r = (l_pad is not None), (r_pad is not None)
    last = in_vectors.shape[-1]
    dsize = in_vectors.size // last

    x = np.empty((last, dsize + _l + _r))
    x[:, 0] = l_pad
    x[:, 1] = r_pad
    x[:, (_l + _r):] = in_vectors.reshape(dsize, last).swapaxes(1, 0)
    x.sort(axis=1)  # then sort

    # pick only few
    if return_reduced:
        if (x.shape[-1] % reduce_to) == 0:
            _redidx = np.s_[::(x.shape[-1] // reduce_to)]
        else:
            _redidx = (np.linspace(0, 1, reduce_to, endpoint=False) * x.shape[-1]).astype(int)
        x = x[:, _redidx]

    # ecdf of x : 0, 1/(w*h) ... (w*h)/(w*h), 1
    if return_cumprob:
        ecdf_x_ = np.empty((dsize + _l + _r))
        if _l:
            ecdf_x_[0] = 0.
        if _r:
            ecdf_x_[-1] = 1.
        ecdf_x_[_l:(dsize + 1)] = np.linspace(1 / dsize, 1., dsize, endpoint=True)

        return x, ecdf_x_

    return x
