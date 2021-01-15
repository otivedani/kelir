import unittest
import numpy as np
from PIL import Image
from synciro import adjustment


class TestThreeway(unittest.TestCase):

    def setUp(self):
        with Image.open('tests/sample_image/test_0.jpg') as img:
            self.imgdata = np.asarray(img)
                        
    def test_mix_channel(self):
        points = [
            [[128, 128]],
            ]
        adjustment.curves(self.imgdata, points)
        
    def test_each_channel(self):
        points = [
            [[0,2], [128,129]],
            [[127,128], [255,253]],
            [[0,2], [128,129], [255,253]]
        ]
        adjustment.curves(self.imgdata, points)

    def test_all_channel(self):
        points = [
            [[0,2], [128,129]],
            [[127,128], [255,253]],
            [[255,253], [0,2], [128,129]],
            [[128,130]], # a bit overall bright
        ]
        adjustment.curves(self.imgdata, points)
    
    def test_ineq_channel(self):
        with self.assertRaises(AssertionError):
            points = [
                [[128,129]],
                [[0,2], [128,129]], # 2 channel
            ]
            adjustment.curves(self.imgdata, points)

    def test_ineq_dimension(self):
        with self.assertRaises(AssertionError):
            points = [
                [[128,129]],
                [[128,129]],
                [[0,2,2], [128,129,1]], # 3D
            ]
            adjustment.curves(self.imgdata, points)
    
    def tearDown(self):
        del self.imgdata


if __name__ == '__main__':
    unittest.main()