import numpy as np
import numpy.typing as npt
from typing import Tuple, Union


BoolNDArray = Union[np.bool_, np.ndarray]


def color_areas(image: np.ndarray,
                normed: bool = False
                ) -> Tuple[BoolNDArray, ...]:
    _image_check(image)
    
    rgb = rgb_areas(image, skip_check=True)
    cmy = cmy_areas(image, skip_check=True)
    w = white_areas(image, normed=normed, skip_check=True)
    n = neutral_areas(image, normed=normed, skip_check=True)
    b = black_areas(image, normed=normed, skip_check=True)
    
    return (*rgb, *cmy, w, n, b)
    

def rgb_areas(image: np.ndarray,
              skip_check: bool = False,
                ) -> Tuple[BoolNDArray, ...]:
    """
    'RGBs' areas of image
    
    Returns
    -------
    ndarray, ndarray, ndarray
        one channel image mask shaped width, height from each red, green, blue
    """
    if not skip_check:
        _image_check(image)
        
    image_max = np.argmax(image, axis=-1)
    return (image_max == 0,
            image_max == 1,
            image_max == 2)


def cmy_areas(image: np.ndarray,
              skip_check: bool = False,
                ) -> Tuple[BoolNDArray, ...]:
    """
    'CMYs (inverse RGBs)' areas of image
    
    Returns
    -------
    ndarray, ndarray, ndarray
        one channel image mask shaped width, height from each cyan, magenta, yellow
    """
    if not skip_check:
        _image_check(image)
        
    image_min = np.argmin(image, axis=-1)
    return (image_min == 0,
            image_min == 1,
            image_min == 2)


def white_areas(image: np.ndarray,
                normed: bool = False,
                skip_check: bool = False,
                ) -> BoolNDArray:
    """
    'White' areas of image

    Parameters
    ----------
    image: ndarray
        image to find the area
    normed: bool, optional
        either image was normed (ranged [0.0, 1.0]) or not (ranged [0, 255]), by default 'False'
    
    Returns
    -------
    ndarray
        one channel image mask shaped width, height
    """
    if not skip_check:
        _image_check(image)

    return np.all(image > 128 if not normed else .5, axis=-1)


def black_areas(image: np.ndarray,
                normed: bool = False,
                skip_check: bool = False,
                ) -> BoolNDArray:
    """
    'Black' areas of image

    Parameters
    ----------
    image: ndarray
        image to find the area
    normed: bool, optional
        either image was normed (ranged [0.0, 1.0]) or not (ranged [0, 255]), by default 'False'
    
    Returns
    -------
    ndarray
        one channel image mask shaped width, height
    """
    if not skip_check:
        _image_check(image)

    return np.all(image < 128 if not normed else .5, axis=-1)


def neutral_areas(image: np.ndarray,
                  normed: bool = False,
                  skip_check: bool = False,
                ) -> BoolNDArray:
    """
    'Neutral' areas of image

    Parameters
    ----------
    image: ndarray
        image to find the area
    normed: bool, optional
        either image was normed (ranged [0.0, 1.0]) or not (ranged [0, 255]), by default 'False'
    
    Returns
    -------
    ndarray
        one channel image mask shaped width, height
    """
    if not skip_check:
        _image_check(image)

    return np.all((image != 0) * (image != 255 if not normed else 1.0), axis=-1)


def _image_check(image: np.ndarray) -> None:
    """
    check whether image is valid

    Parameters
    ----------
    image: ndarray
        image to be checked

    Raise
    -----
    ValueError if image shape is less than 2 and not have 3 channel 
    """
    if (len(image.shape) < 2) or (image.shape[-1] != 3):
        raise ValueError("Image must be in shape (Any, 3)")

    return