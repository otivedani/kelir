import numpy as np
from PIL import Image
from typing import Union
from .. import adjustment, statistic


def hist_match(source: Union[np.ndarray, Image.Image],
               reference: Union[np.ndarray, Image.Image],
               to_lut: bool = False,
               copy: bool = True,
               precision: int = 256,
               ) -> Union[np.ndarray, Image.Image]:
    """Match each channel histogram of two PIL.Image or ndarray"""
    if type(source) is not type(reference):
        raise TypeError('source and reference must both either ndarray or PIL.Image')

    _source = source.copy() if copy else source

    if isinstance(source, Image.Image):
        if _source.mode != reference.mode:
            _source = _source.convert(reference.mode)
        _source = np.asarray(_source)

    _refere = np.asarray(reference)

    # safety countermeasures for exceeding precision
    _precision = min(precision, _source[..., 0].size, _refere[..., 0].size)

    x_src = statistic.ecdf(_source, reduce_to=_precision)
    x_ref = statistic.ecdf(_refere, reduce_to=_precision)

    points = [np.dstack(qsqt).squeeze() for qsqt in zip(x_src, x_ref)]

    # apply curves based of points, write in place if possible (to copy or not copy have been decided before).
    target = np.ones((256, 3)) * np.arange(0, 256)[:, None] if to_lut else _source
    adjustment.curves(target, points, copy=(not target.flags.writeable))

    if isinstance(source, Image.Image) and not to_lut:
        target = Image.fromarray(target, mode=reference.mode).convert(source.mode)

    return target
