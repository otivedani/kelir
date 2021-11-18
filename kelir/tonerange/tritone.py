import numpy as np
import numpy.typing as npt
from typing import Tuple

# simple separation
# TODO(otivedani) apply gamma for perceptive brightness / gray mode
# TODO(otivedani) grayscale mode to module
# TODO(otivedani) overlap options, blur, mask or applied

RGB2L_METHODS = {
            'L': [0.299, 0.587, 0.114],  # ITU-R 601-2 Luma
            'L2': [0.2126, 0.7152, 0.0722],
            'Avg': [1/3, 1/3, 1/3],
        }


# TODO(otivedani) : to be deprecated, remain for backward compatibility
def split_tritone(image, 
                  midrange: Tuple[int, int] = (70, 180), 
                  luma_method: str = 'L', 
                  c: float = 1):
    """Three-way tone range : Shadows, Midtones, Highlights

    Assuming last dimension of image is the channel.
    """
    if luma_method not in RGB2L_METHODS.keys():
        raise ValueError(f'Unknown luma conversion method "{luma_method}"')

    n_channel = image.shape[-1]
    _in_array = image.reshape(-1, n_channel)

    if n_channel == 1:
        img_L = _in_array[:, 0]
    elif n_channel == 3:
        lc = RGB2L_METHODS[luma_method]
        img_L = (lc[0]*(_in_array[:, 0]**c) +
                 lc[1]*(_in_array[:, 1]**c) +
                 lc[2]*(_in_array[:, 2]**c))**(1/c)
    else:
        raise ValueError('image channel must be either 1 or 3')

    p, q = midrange

    shadow_mask = img_L < p
    midtones_mask = (img_L >= p) * (img_L < q)
    highlight_mask = img_L >= q

    return shadow_mask.reshape(image.shape[:-1]),\
        midtones_mask.reshape(image.shape[:-1]),\
        highlight_mask.reshape(image.shape[:-1])


def _split(image: np.ndarray,
           midrange: npt.ArrayLike = ((70, 180)),
           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split each last dimension of image into three by the midtones

    if supplied midrange is 1d, then it is broadcasted.
    """
    midrange = np.asarray(midrange)
    # when input is one pair, assume it is array of pair with 1 length.
    if len(midrange.shape) == 1:
        midrange = midrange[None, :]

    lo = image < midrange[:, 0]
    md = (image >= midrange[:, 0]) * (image < midrange[:, 1])
    hi = image >= midrange[:, 1]

    return lo, md, hi
