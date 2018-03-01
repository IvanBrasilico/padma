# Tescases for padma.py

import json
import os
import unittest
from io import BytesIO
from threading import Thread

from padma.app import app, classify_process

TEST_IMAGE = os.path.join(os.path.dirname(__file__), 'test.png')
CHEIO_IMAGE = os.path.join(os.path.dirname(__file__), 'cheio.jpg')
VAZIO_IMAGE = os.path.join(os.path.dirname(__file__), 'vazio.jpg')
STAMP_IMAGE = os.path.join(os.path.dirname(__file__), 'stamp1.jpg')


class FlaskTestCase(unittest.TestCase):

    def setUp(self):
        t = Thread(target=classify_process, args=())
        t.daemon = True
        t.start()
        app.testing = True
        self.app = app.test_client()

    def tearDown(self):
        """Limpa o ambiente"""
        pass

    """def test_prediction_ResNet(self):
        image = open(TEST_IMAGE, 'rb').read()
        data = {}
        data['image'] = (BytesIO(image), 'image')
        rv = self.app.post(
            '/predict?model=resnet',
            content_type='multipart/form-data', data=data)
        test_dict = json.loads(rv.data.decode())
        assert test_dict.get('success') is not None
        assert test_dict.get('success') is True
        assert test_dict.get('predictions')[0].get('label') == 'beagle'
        assert b'beagle' in rv.data
    """

    def test_prediction_Vazio(self):
        image = open(VAZIO_IMAGE, 'rb').read()
        data = {}
        data['image'] = (BytesIO(image), 'image')
        rv = self.app.post(
            '/predict?model=vazio',
            content_type='multipart/form-data', data=data)
        test_dict = json.loads(rv.data.decode())
        assert test_dict.get('success') is not None
        assert test_dict.get('success') is True
        assert test_dict.get('predictions') is not None
        print(test_dict.get('predictions'))
        assert test_dict.get('predictions')[0].get('0') > 0.5
        assert b'"1"' in rv.data

    def test_prediction_Cheio(self):
        image = open(CHEIO_IMAGE, 'rb').read()
        data = {}
        data['image'] = (BytesIO(image), 'image')
        rv = self.app.post(
            '/predict?model=vazio',
            content_type='multipart/form-data', data=data)
        test_dict = json.loads(rv.data.decode())
        assert test_dict.get('success') is not None
        assert test_dict.get('success') is True
        assert test_dict.get('predictions') is not None
        print(test_dict.get('predictions'))
        assert test_dict.get('predictions')[0].get('1') > 0.5
        assert b'"1"' in rv.data

    def test_naive(self):
        image = open(STAMP_IMAGE, 'rb').read()
        data = {}
        data['image'] = (BytesIO(image), 'image')
        rv = self.app.post(
            '/predict?model=naive',
            content_type='multipart/form-data', data=data)
        preds = json.loads(rv.data.decode())
        preds = preds['predictions']
        print(preds)
        assert preds['class'] == 'cc'
        assert abs(preds['bbox'][0] - 227) < 4
        assert abs(preds['bbox'][1] - 28) < 4
        assert abs(preds['bbox'][2] - 472) < 4
        assert abs(preds['bbox'][3] - 206) < 4
