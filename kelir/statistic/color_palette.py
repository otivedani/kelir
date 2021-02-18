import numpy as np


def make_palette(image: np.ndarray,
                 n_palette: int = 256
                 ) -> np.ndarray:
    """Create color palette from image using median cut algorithm"""
    e_factor = np.log2(n_palette)  # 2^x
    if not e_factor.is_integer():
        raise ValueError("Number of palette must a result of exponential")

    # create image copy, reduce so it is divisible
    palette_image = image.copy().reshape(1, -1, image.shape[-1])
    factored_length = palette_image.shape[-2] - (palette_image.shape[-2] % n_palette)
    palette_image = palette_image[:, :factored_length, :]

    # recursive sort, median cut algorithm
    for npal in np.exp2(np.arange(e_factor + 1)).astype(int):
        longest_ch = (palette_image.max(axis=-2) - palette_image.min(axis=-2)).argmax(axis=-1)
        for i, l in enumerate(longest_ch):
            palette_image[i, :] = palette_image[i, palette_image[i, :, l].argsort()]
        palette_image = palette_image.reshape(npal, -1, palette_image.shape[-1])

    _base = 255. if image.dtype is np.uint8 else 1.
    palette_image = ((palette_image / _base).mean(axis=-2) * _base)

    return palette_image.astype(image.dtype)
