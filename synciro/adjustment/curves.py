import numpy as np
from scipy.interpolate import interp1d

"""
Apply curves to image

param :
    - image : numpy array, with shape (width, height, 3)
    - points : list of each channel curve points (x, y)
        ~ len(points) must be either :
            - 1 (equal adjust to all channel)
            - 3 (respective adjust to each channel)
            - 4 (respective adjust each and all channel)
        ~ len(points[i]) keep it < 16 for best result
        ~ len(points[i][j]) is a pair of points [x, y]
"""


def adjust_curves(image, points, mode='precise'):
    _CURVES_STRATS_ = {1, 3, 4}
    _CURVES_MODES_ = {
        'precise': _fit_precise,  # unclosed, fill with first-last
        'closed': _fit_closed,  # closed, fill with linear
        'extrapolate': _fit_extrapolate,  # unclosed, fill with extrapolation
    }

    strat = len(points)
    assert(strat in _CURVES_STRATS_)
    assert(mode in _CURVES_MODES_.keys())

    # points to interpolate
    curves_f = [_CURVES_MODES_[mode](p) for p in points]

    # prepare output
    out_image = image.copy()

    # case 3 : apply to each channel separately (c0, c1, c2)->(R, G, B)
    # case 1 : apply to all channel equally (c0)->(RGB)
    # case 4 : apply 3 + 1 (c0, c1, c2)->(R, G, B) ; (c3)->(RGB)
    _apply_each = strat in {3, 4}
    _apply_all = strat in {1, 4}

    if (_apply_each):
        for i, curve_f in enumerate(curves_f[:3]):
            out_image[:, :, i] = np.clip(curve_f(out_image[:, :, i]), 0, 255)

    if (_apply_all):
        out_image = np.clip(curves_f[-1](out_image), 0, 255)

    # applied curves
    return out_image.astype(image.dtype)


def _prep_pairs(p, close_range=False):
    # create and check
    pt = np.asarray(p, dtype=np.uint8)
    assert(pt.shape[-1] == 2)
    # sort and remove duplicates in x
    _pt, i = np.unique(pt[:, 0], return_index=True)
    pt = pt[i]

    if close_range:  # begin with 0, end with 255
        if _pt[0] != 0:
            pt = np.vstack(([0, 0], pt))
        if _pt[-1] != 255:
            pt = np.vstack((pt, [255, 255]))

    return pt


# unclosed, fill with first-last
def _fit_precise(points):
    pt_v = _prep_pairs(points, close_range=False)
    return interp1d(pt_v[:, 0], pt_v[:, 1],
                    kind=('quadratic' if pt_v.shape[0] > 2 else 'linear'),
                    bounds_error=False, fill_value=(pt_v[0, 1], pt_v[-1, 1]))


# closed, fill with linear
def _fit_closed(points):
    pt_v = _prep_pairs(points, close_range=True)
    return interp1d(pt_v[:, 0], pt_v[:, 1],
                    kind=('quadratic' if pt_v.shape[0] > 2 else 'linear'),
                    bounds_error=False, fill_value=(pt_v[0, 1], pt_v[-1, 1]))


# unclosed, fill with extrapolation
def _fit_extrapolate(points):
    pt_v = _prep_pairs(points, close_range=False)
    return interp1d(pt_v[:, 0], pt_v[:, 1],
                    kind=('quadratic' if pt_v.shape[0] > 2 else 'linear'),
                    fill_value='extrapolate')
