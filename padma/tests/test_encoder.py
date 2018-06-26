# Tescases for models/vazios/vazios.py
import os
import unittest

from PIL import Image

from padma.models.encoder.encoder import EncoderModel

VAZIO_IMAGE = os.path.join(os.path.dirname(__file__), 'vazio.jpg')
CHEIO_IMAGE = os.path.join(os.path.dirname(__file__), 'cheio.jpg')


class TestModel(unittest.TestCase):
    def setUp(self):
        self.vazio = Image.open(VAZIO_IMAGE)
        self.cheio = Image.open(CHEIO_IMAGE)
        self.model = EncoderModel()

    def tearDown(self):
        pass

    def test_encode(self):
        codev = self.model.predict(self.vazio)
        assert codev is not None
        codec = self.model.predict(self.cheio)
        assert codec is not None
        assert codec != codev
