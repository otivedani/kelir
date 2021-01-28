import unittest
import numpy as np
from PIL import Image
from synciro import features

class TestStats(unittest.TestCase):

    def setUp(self):
        with Image.open('tests/sample_image/test_0.jpg') as img:
            self.imgdata = np.asarray(img)
                        
    def test_stats(self):
        x, y = features.ecdf(self.imgdata)
        self.assertEqual(y.shape[0], self.imgdata.shape[0]*self.imgdata.shape[1])
        self.assertEqual(x.shape[-1], self.imgdata.shape[0]*self.imgdata.shape[1])
        self.assertEqual(x.shape[0], self.imgdata.shape[-1])
        self.assertTrue(np.all(np.min(x, axis=1) == self.imgdata.min(axis=(0,1))))
        self.assertTrue(np.all(np.max(x, axis=1) == self.imgdata.max(axis=(0,1))))

        x, y = features.ecdf(self.imgdata[:,:,:], extend=True)
        self.assertTrue(np.all(np.min(x, axis=1) == 0))
        self.assertTrue(np.all(np.max(x, axis=1) == 255))
        

        with self.assertRaises(ValueError):
            features.ecdf(self.imgdata.ravel())
    
    def tearDown(self):
        del self.imgdata


if __name__ == '__main__':
    unittest.main()