from kelir import statistic
import numpy as np
from PIL import Image
import unittest


class TestEcdf(unittest.TestCase):

    def setUp(self):
        with Image.open('test/sample_image/test_0.jpg') as img:
            self.imgdata = np.asarray(img)

    def test_stats(self):
        x, y = statistic.ecdf(self.imgdata, return_cumprob=True)
        self.assertEqual(y.shape[0], self.imgdata.shape[0] * self.imgdata.shape[1])
        self.assertEqual(x.shape[-1], self.imgdata.shape[0] * self.imgdata.shape[1])
        self.assertEqual(x.shape[0], self.imgdata.shape[-1])
        self.assertTrue(np.all(np.min(x, axis=1) == self.imgdata.min(axis=(0, 1))))
        self.assertTrue(np.all(np.max(x, axis=1) == self.imgdata.max(axis=(0, 1))))

        x_2 = statistic.ecdf(self.imgdata[:, :, :], l_pad=-1111, r_pad=99999)
        self.assertTrue(np.all(np.min(x_2, axis=1) == -1111))
        self.assertTrue(np.all(np.max(x_2, axis=1) == 99999))

        with self.assertRaises(ValueError):
            statistic.ecdf(self.imgdata.ravel())

    def test_reduced_stats(self):
        for u in (98, 99, 100, 101, 107, 3584, 3585, 3586, 3587, 3588, 3589, 3590):
            x = statistic.ecdf(self.imgdata, reduce_to=u)
            self.assertEqual(x.shape[-1], u)

        with self.assertRaises(ValueError):
            statistic.ecdf(self.imgdata, reduce_to=9999999999)

    def tearDown(self):
        del self.imgdata


if __name__ == '__main__':
    unittest.main()
