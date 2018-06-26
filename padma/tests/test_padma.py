# Tescases for padma.py

import json
import os
import unittest
from io import BytesIO
from threading import Thread

from padma.app import app
from padma.modelserver import classify_process

from ajna_commons.flask.login import DBUser

TEST_IMAGE = os.path.join(os.path.dirname(__file__), 'test.png')
CHEIO_IMAGE = os.path.join(os.path.dirname(__file__), 'cheio.jpg')
VAZIO_IMAGE = os.path.join(os.path.dirname(__file__), 'vazio.jpg')
STAMP_IMAGE = os.path.join(os.path.dirname(__file__), 'stamp1.jpg')

t = Thread(target=classify_process, args=())
t.daemon = True
t.start()

class FlaskTestCase(unittest.TestCase):

    def setUp(self):
        app.testing = True
        self.app = app.test_client()
        DBUser.dbsession = None  # Bypass mongodb authentication
        # A linha acima faz com que qqer usuario=senha com
        # conteúdo igual (A=A) autentique.

    def tearDown(self):
        """Limpa o ambiente"""
        rv = self.logout()
        assert rv is not None

    def get_token(self, url):
        response = self.app.get(url, follow_redirects=True)
        self.csrf_token = str(response.data)
        begin = self.csrf_token.find('csrf_token"') + 10
        end = self.csrf_token.find('username"') - 10
        self.csrf_token = self.csrf_token[begin: end]
        # print('token****', self.csrf_token)
        begin = self.csrf_token.find('value="') + 7
        end = self.csrf_token.find('/>')
        self.csrf_token = self.csrf_token[begin: end]
        # print('token****', self.csrf_token)

    def login(self, username, senha, next_url=''):
        url = '/login'
        if next_url:
            url = url + '?next=' + next_url
        self.get_token(url)
        return self.app.post(url, data=dict(
            username=username,
            senha=senha,
            csrf_token=self.csrf_token
        ), follow_redirects=True)

    def logout(self):
        return self.app.get('/logout',
                            data=dict(csrf_token=self.csrf_token),
                            follow_redirects=True)

    def test_login_invalido(self):
        rv = self.login('none', 'error')
        print(rv)
        assert rv is not None
        assert b'401' in rv.data

    def test_index(self):
        rv = self.login('ajna', 'ajna')
        assert rv is not None

        rv = self.app.get('/', follow_redirects=True)
        assert b'AJNA' in rv.data

    def test_tela_teste(self):
        self.login('ajna', 'ajna')
        image = open(STAMP_IMAGE, 'rb').read()
        data = {}
        data['file'] = (BytesIO(image), 'image')
        self.get_token('/teste')
        data['csrf_token'] = self.csrf_token
        rv = self.app.post(
            '/teste',
            content_type='multipart/form-data', data=data)
        print(rv.data)
        assert rv.data is not None
        assert b'Testar modelos' in rv.data

    def test_prediction_Vazio(self):
        self.login('ajna', 'ajna')
        image = open(VAZIO_IMAGE, 'rb').read()
        data = {}
        data['image'] = (BytesIO(image), 'image')
        rv = self.app.post(
            '/predict?model=vazio',
            content_type='multipart/form-data', data=data)
        print(rv)
        test_dict = json.loads(rv.data.decode())
        assert test_dict.get('success') is not None
        assert test_dict.get('success') is True
        assert test_dict.get('predictions') is not None
        print(test_dict.get('predictions'))
        assert test_dict.get('predictions')[0].get('0') > 0.5
        assert b'"1"' in rv.data

    def test_prediction_Cheio(self):
        self.login('ajna', 'ajna')
        image = open(CHEIO_IMAGE, 'rb').read()
        data = {}
        data['image'] = (BytesIO(image), 'image')
        rv = self.app.post(
            '/predict?model=vazio',
            content_type='multipart/form-data', data=data)
        print(rv)
        test_dict = json.loads(rv.data.decode())
        assert test_dict.get('success') is not None
        assert test_dict.get('success') is True
        assert test_dict.get('predictions') is not None
        print(test_dict.get('predictions'))
        assert test_dict.get('predictions')[0].get('1') > 0.5
        assert b'"1"' in rv.data

    def test_naive(self):
        self.login('ajna', 'ajna')
        image = open(STAMP_IMAGE, 'rb').read()
        data = {}
        data['image'] = (BytesIO(image), 'image')
        rv = self.app.post(
            '/predict?model=naive',
            content_type='multipart/form-data', data=data)
        preds = json.loads(rv.data.decode())
        preds = preds['predictions']
        print(preds)
        preds = preds[0]
        assert preds['class'] == 'cc'
        assert abs(preds['bbox'][0] - 25) < 5
        assert abs(preds['bbox'][1] - 226) < 5
        assert abs(preds['bbox'][2] - 204) < 5
        assert abs(preds['bbox'][3] - 472) < 5

    def test_prediction_Vazio2(self):
        self.login('ajna', 'ajna')
        image = open(VAZIO_IMAGE, 'rb').read()
        data = {}
        data['image'] = (BytesIO(image), 'image')
        rv = self.app.post(
            '/predict?model=vaziosvm',
            content_type='multipart/form-data', data=data)
        test_dict = json.loads(rv.data.decode())
        assert test_dict.get('success') is not None
        assert test_dict.get('success') is True
        assert test_dict.get('predictions') is not None
        print(test_dict.get('predictions'))
        assert test_dict.get('predictions')[0].get('vazio') is True

    def test_prediction_Cheio2(self):
        self.login('ajna', 'ajna')
        image = open(CHEIO_IMAGE, 'rb').read()
        data = {}
        data['image'] = (BytesIO(image), 'image')
        rv = self.app.post(
            '/predict?model=vaziosvm',
            content_type='multipart/form-data', data=data)
        print(rv.data)
        test_dict = json.loads(rv.data.decode())
        assert test_dict.get('success') is not None
        assert test_dict.get('success') is True
        assert test_dict.get('predictions') is not None
        print(test_dict.get('predictions'))
        assert test_dict.get('predictions')[0].get('vazio') is False
