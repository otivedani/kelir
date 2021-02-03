import numpy as np
from PIL import Image
from typing import Union
from .. import adjustment, statistic


def hist_match(source: Union[np.ndarray, Image.Image],
               reference: Union[np.ndarray, Image.Image],
               to_lut: bool = False,
               copy: bool = True,
               ) -> Union[np.ndarray, Image.Image]:
    """Match each channel histogram of two PIL.Image or ndarray"""
    assert(type(source) == type(reference))

    target = source.copy() if copy else source

    if isinstance(source, Image.Image):
        if target.mode != reference.mode:
            target = target.convert(reference.mode)

    src_img = np.asarray(target)
    ref_img = np.asarray(reference)

    x_src, y_src = statistic.ecdf(src_img)
    x_ref, y_ref = statistic.ecdf(ref_img)

    samples_src = np.asarray([x_src[:, q] for q in _sample_vector(len(y_src), precision=256)]).swapaxes(1, 0)
    samples_ref = np.asarray([x_ref[:, q] for q in _sample_vector(len(y_ref), precision=256)]).swapaxes(1, 0)

    points = [np.dstack(qsqt).squeeze() for qsqt in zip(samples_src, samples_ref)]

    if to_lut:
        target = np.ones((256, 3)) * np.arange(0, 256)[:, None]

    adjustment.curves(target, points, copy=False)

    return target


def _sample_vector(x: int, precision: int = 256) -> np.ndarray:
    return (np.linspace(0, 1, precision, endpoint=False) * x).astype(int)
