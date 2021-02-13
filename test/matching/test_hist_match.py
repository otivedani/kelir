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
        # numpy array, return a copy
        img_result_0 = matching.hist_match(self.img_source, self.img_target)
        self.assertFalse(np.shares_memory(img_result_0, self.img_source))

        img_result_2 = matching.hist_match(self.img_source, self.img_source)
        self.assertIsNot(self.img_source, img_result_2)

        matching.hist_match(self.img_source, self.img_target, precision=999999999)  # safe
        matching.hist_match(self.img_source, self.img_target, precision=257)
        matching.hist_match(self.img_source, self.img_target, precision=259)
        matching.hist_match(self.img_source, self.img_target, precision=261)

    def test_nocopy(self):
        # numpy array, return same data
        _img_source = self.img_source.copy()
        img_result_0 = matching.hist_match(_img_source, self.img_target, copy=False)
        self.assertTrue(np.shares_memory(img_result_0, _img_source))

        img_result_2 = matching.hist_match(_img_source, _img_source, copy=False)
        self.assertIs(img_result_2, _img_source)
        del _img_source

    def test_lut1d(self):
        # numpy array, return LUT
        lut1d = matching.hist_match(self.img_source, self.img_target, to_lut=True)
        self.assertEqual(np.asarray(lut1d).shape, (256, 3))

    def test_typemismatch(self):
        with self.assertRaises(TypeError):
            matching.hist_match(Image.fromarray(self.img_source), self.img_target)

    def tearDown(self):
        del self.img_source
        del self.img_target


if __name__ == '__main__':
    unittest.main()
