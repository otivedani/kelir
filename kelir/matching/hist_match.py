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
    assert(type(source) == type(reference))

    _source = source.copy() if copy else source

    if isinstance(source, Image.Image):
        if _source.mode != reference.mode:
            _source = _source.convert(reference.mode)
        _source = np.asarray(_source)

    _refere = np.asarray(reference)

    x_src, y_src = statistic.ecdf(_source)
    x_ref, y_ref = statistic.ecdf(_refere)

    samples_src = np.asarray([x_src[:, q] for q in _sample_vector(len(y_src), precision=precision)]).swapaxes(1, 0)
    samples_ref = np.asarray([x_ref[:, q] for q in _sample_vector(len(y_ref), precision=precision)]).swapaxes(1, 0)

    points = [np.dstack(qsqt).squeeze() for qsqt in zip(samples_src, samples_ref)]

    # apply curves based of points, write in place if possible (to copy or not copy have been decided before).
    target = np.ones((256, 3)) * np.arange(0, 256)[:, None] if to_lut else _source
    adjustment.curves(target, points, copy=(not target.flags.writeable))

    if isinstance(source, Image.Image) and not to_lut:
        target = Image.fromarray(target, mode=reference.mode).convert(source.mode)

    return target


def _sample_vector(x: int, precision: int = 256) -> np.ndarray:
    return (np.linspace(0, 1, precision, endpoint=False) * x).astype(int)
