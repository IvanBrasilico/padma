# Tescases for models/vazios/vazios.py
import os
import unittest

from keras.preprocessing.image import img_to_array
from PIL import Image

from padma.models.bbox.bbox import NaiveModel

IMAGE = os.path.join(os.path.dirname(__file__), 'stamp1.jpg')


class TestModel(unittest.TestCase):
    def setUp(self):
        self.image = img_to_array(Image.open(IMAGE))
        self.naive = NaiveModel()
        # self.retina = RetinaModel()

    def tearDown(self):
        pass

    def test_naive(self):
        preds = self.naive.predict(self.image)
        print(preds)
        assert preds['class'] == 'cc'
        assert abs(preds['bbox'][0] - 227) < 4
        assert abs(preds['bbox'][1] - 28) < 4
        assert abs(preds['bbox'][2] - 472) < 4
        assert abs(preds['bbox'][3] - 206) < 4

    """def test_resnet(self):
        preds = self.retina.predict(self.image)
        print(preds)
        print(preds)
        assert 'cc' in preds[0][1]
        assert abs(preds[0][0][0] - 227) < 10
        assert abs(preds[0][0][1] - 28) < 10
        assert abs(preds[0][0][2] - 472) < 10
        assert abs(preds[0][0][3] - 206) < 10
    """


if __name__ == '__main__':
    unittest.main()