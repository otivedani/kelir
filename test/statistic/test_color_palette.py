from kelir import statistic
import numpy as np
from PIL import Image
import unittest


class TestColorPalette(unittest.TestCase):

    def setUp(self):
        with Image.open('test/sample_image/test_0.jpg') as img:
            self.imgdata = np.asarray(img)

    def test_default(self):
        palette_256 = statistic.make_palette(self.imgdata)
        self.assertTrue(palette_256.shape == (256, self.imgdata.shape[-1]))
        palette_16 = statistic.make_palette(self.imgdata, n_palette=16)
        self.assertTrue(palette_16.shape == (16, self.imgdata.shape[-1]))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            statistic.make_palette(self.imgdata, n_palette=9)

    def tearDown(self):
        del self.imgdata


if __name__ == '__main__':
    unittest.main()
