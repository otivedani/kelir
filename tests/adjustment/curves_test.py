import numpy as np
from PIL import Image
from synciro import adjustment
import unittest


class TestCurves(unittest.TestCase):

    def setUp(self):
        self.points = [
            [[0, 2], [128, 129], [255, 255]],
            [[255, 200], [0, 2], [74, 39]],
            [[0, 0], [64, 64], [128, 150], [72, 72], [200, 200]],
            [[5, 10], [250, 230]],
        ]

        with Image.open('tests/sample_image/test_0.jpg') as image:
            self.image = np.asarray(image)

    def test_all_channel(self):
        # 1 curve to all
        resimg_1 = adjustment.curves(self.image, self.points[-1:], mode='precise')
        self.assertTrue(np.all(resimg_1[self.image == 250] == 230))  # 250 -> 230
        self.assertTrue(np.all(resimg_1.max(axis=(0, 1)) == 230))  # result of clipping

        # 3 curves to each
        resimg_3 = adjustment.curves(self.image, self.points[:-1], mode='precise')
        self.assertTrue(np.all(resimg_3[:, :, 0][self.image[:, :, 0] == 0] == 2))  # 0 -> 2
        self.assertTrue(np.all(resimg_3[:, :, 1][self.image[:, :, 1] == 74] == 39))  # 74 -> 39
        self.assertTrue(np.all(resimg_3[:, :, 2].max(axis=(0, 1)) == 200))  # result of clipping

        # 3 curves to each, 1 to all
        resimg_4 = adjustment.curves(self.image, self.points, mode='precise')
        resimg_3_1 = adjustment.curves(resimg_3, self.points[-1:], mode='precise')
        self.assertTrue(np.all(resimg_3_1 == resimg_4))  # 3+1 == 4

        # closed mode : pad with 0,0 and 255,255
        resimg_1c = adjustment.curves(self.image, self.points[-1:], mode='closed')
        self.assertTrue(np.all(resimg_1c[self.image == 0] == 0))
        self.assertTrue(np.all(resimg_1c[self.image == 255] == 255))

    def test_ineq_channel(self):
        with self.assertRaises(AssertionError):
            adjustment.curves(self.image, self.points[:2])

    def test_ineq_dimension(self):
        with self.assertRaises(AssertionError):
            points = [
                [[0, 2, 2], [128, 129, 1]],  # 3D
            ]
            adjustment.curves(self.image, points)

    def test_unique_points(self):
        points = [
            [[128, 129]],
            [[128, 129], [128, 170]],
            [[0, 2], [0, 9], [128, 129]],
        ]
        for pt in points:
            _pt = adjustment._prep_pairs(pt)
            self.assertTrue(np.any(
                np.unique(_pt[:, 0], return_counts=True)[1] == 1))

        with self.assertRaises(ValueError):
            adjustment._prep_pairs(points)

    def tearDown(self):
        del self.image


if __name__ == '__main__':
    unittest.main()
