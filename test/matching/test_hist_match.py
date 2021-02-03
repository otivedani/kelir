from kelir import matching
import numpy as np
from PIL import Image
import unittest


# TODO(otivedani) image color metrics (maybe SSIM?)
class TestMatch_Hist(unittest.TestCase):

    def setUp(self):
        with Image.open('test/sample_image/test_0.jpg') as img:
            self.img_source = np.asarray(img)
        with Image.open('test/sample_image/test_1.jpg') as img:
            self.img_target = np.asarray(img)

    def test_default(self):
        img_result = matching.hist_match(self.img_source, self.img_target)
        self.assertFalse(np.shares_memory(img_result, self.img_source))

    def test_nocopy(self):
        _img_source = self.img_source.copy()
        img_result = matching.hist_match(_img_source, self.img_target, copy=False)
        self.assertTrue(np.shares_memory(img_result, _img_source))
        del _img_source

    def test_lut1d(self):
        lut1d = matching.hist_match(self.img_source, self.img_target, to_lut=True)
        self.assertEqual(np.asarray(lut1d).shape, (256, 3))

    def tearDown(self):
        del self.img_source
        del self.img_target


if __name__ == '__main__':
    unittest.main()
