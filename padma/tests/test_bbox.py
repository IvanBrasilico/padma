# Tescases for models/vazios/vazios.py
import os
import unittest

from PIL import Image

from padma.models.bbox.bbox import NaiveModel
from padma.models.conteiner20e40.bbox import SSDMobileModel

IMAGE = os.path.join(os.path.dirname(__file__), 'stamp1.jpg')


class TestModel(unittest.TestCase):
    def setUp(self):
        self.image = Image.open(IMAGE)
        self.naive = NaiveModel()
        self.ssd = SSDMobileModel()

    def tearDown(self):
        pass

    def test_naive(self):
        preds = self.naive.predict(self.image)
        print(preds)
        preds = preds[0]
        assert preds['class'] == 'cc'
        assert abs(preds['bbox'][0] - 227) < 4
        assert abs(preds['bbox'][1] - 28) < 4
        assert abs(preds['bbox'][2] - 472) < 4
        assert abs(preds['bbox'][3] - 206) < 4

    def test_ssd(self):
        preds = self.ssd.predict(self.image)
        print(preds)
        preds = preds[0]
        assert preds['class'] == 2
        assert abs(preds[0][0][0] - 25) < 5
        assert abs(preds[0][0][1] - 226) < 5
        assert abs(preds[0][0][2] - 204) < 5
        assert abs(preds[0][0][3] - 477) < 5


if __name__ == '__main__':
    unittest.main()
