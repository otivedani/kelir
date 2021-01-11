import unittest
import numpy as np
from PIL import Image
from synciro import tonalrange


class TestThreeway(unittest.TestCase):

    def setUp(self):
        with Image.open('tests/sample_image/test_0.jpg') as img:
            self.imgdata = np.asarray(img)
                        
    def test_default(self):
        sh, md, hl = tonalrange.threeway(self.imgdata)
        self.assertTrue(np.all(sh + md + hl))
        
    def test_midrange(self):
        with self.assertRaises(TypeError):
            self.image_tonemasks = tonalrange.threeway(self.imgdata, midrange=(20))
    
    def tearDown(self):
        del self.imgdata


if __name__ == '__main__':
    unittest.main()