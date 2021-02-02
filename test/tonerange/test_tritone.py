import numpy as np
from PIL import Image
from kelir import tonerange
import unittest


class Testsplit_tritone(unittest.TestCase):

    def setUp(self):
        with Image.open('test/sample_image/test_0.jpg') as img:
            self.imgdata = np.asarray(img)

    def test_default(self):
        sh, md, hl = tonerange.split_tritone(self.imgdata)
        self.assertTrue(np.all(sh + md + hl))
        self.assertEqual(sh.shape, self.imgdata.shape[:-1])
        self.assertEqual(md.shape, self.imgdata.shape[:-1])
        self.assertEqual(hl.shape, self.imgdata.shape[:-1])

        sh, md, hl = tonerange.split_tritone(self.imgdata.mean(axis=(0, 1)))
        self.assertTrue(np.all(sh + md + hl))

    def test_midrange(self):
        with self.assertRaises(TypeError):
            tonerange.split_tritone(self.imgdata, midrange=(20))

    def test_channelmismatch(self):
        with self.assertRaises(ValueError):
            tonerange.split_tritone(self.imgdata[:, :, 1:], midrange=(20, 30))

    def test_splitthree(self):
        # result test
        masks = tonerange._split(self.imgdata)
        self.assertIsInstance(masks, tuple)
        self.assertEqual(len(masks), 3)
        self.assertEqual(masks[0].shape, self.imgdata.shape)

        # valid samples
        tonerange._split(self.imgdata, midrange=(70, 125))
        tonerange._split(self.imgdata[:, :, 0], midrange=((70, 125)))
        tonerange._split(self.imgdata, midrange=((70, 125), (80, 135), (90, 145)))
        tonerange._split(self.imgdata[:, :, 1:], midrange=((70, 125), (80, 135)))

        # error
        with self.assertRaises(ValueError):
            tonerange._split(self.imgdata, midrange=((70, 125), (80, 135)))
        with self.assertRaises(ValueError):
            tonerange._split(self.imgdata[:, :, 1:], midrange=((70, 125), (80, 135), (90, 145)))

        case_a = tonerange._split(self.imgdata, midrange=((70, 125), (80, 135), (90, 145)))
        case_b = tonerange._split(self.imgdata.reshape(-1, self.imgdata.shape[-1]), midrange=((70, 125), (80, 135), (90, 145)))
        case_c = tonerange._split(self.imgdata.reshape(10, self.imgdata.shape[0]//10, self.imgdata.shape[1], self.imgdata.shape[2], ), midrange=((70, 125)))
        for a, b in zip(case_a, case_b):
            self.assertTrue(np.all(a.ravel() == b.ravel()))
        self.assertTrue(np.all(case_a[0][:, :, 0].ravel() == case_c[0][:, :, :, 0].ravel()))

    def tearDown(self):
        del self.imgdata


if __name__ == '__main__':
    unittest.main()
