from kelir import tonerange
import numpy as np
from PIL import Image
import unittest


class Test_RGB(unittest.TestCase):

    def test_default(self):
        white_img = np.ones((32,32,3))*255
        self.assertTrue(tonerange.white_areas(white_img).sum() == 32*32)

        black_img = np.zeros((32,32,3))
        self.assertTrue(tonerange.black_areas(black_img).sum() == 32*32)

        neutral0_img = np.ones((32,32,3))
        neutral1_img = np.ones((32,32,3))*254
        self.assertTrue(tonerange.neutral_areas(neutral0_img).sum() == 32*32)
        self.assertTrue(tonerange.neutral_areas(neutral1_img).sum() == 32*32)
        
        self.assertTrue(tonerange.neutral_areas(white_img).sum() == 0)
        self.assertTrue(tonerange.neutral_areas(black_img).sum() == 0)
        
        invalid1_img = np.zeros((32,32,4))
        with self.assertRaises(ValueError):
            tonerange.rgb_areas(invalid1_img)
        
        invalid2_img = np.zeros((32))
        with self.assertRaises(ValueError):
            tonerange.cmy_areas(invalid2_img)

        invalid3_img = np.ones((32))
        with self.assertRaises(ValueError):
            tonerange.color_areas(invalid3_img)


if __name__ == '__main__':
    unittest.main()
