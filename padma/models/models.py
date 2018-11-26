import os
from PIL import Image
from sklearn.externals import joblib

from padma.models.bbox.bbox import NaiveModel
from padma.models.conteiner20e40.bbox import SSDMobileModel
from padma.models.encoder.encoder import EncoderModel
from padma.models.peso.peso import PesoModel
from padma.models.peso.peso2 import PesoModel2
from padma.models.vazios.vazio2 import VazioSVMModel
from padma.models.vazios.vazios import VazioModel


class BaseModel():
    def __init__(self, joblib_file=None):
        if (joblib_file is not None) and os.path.exists(joblib_file):
            print('*Loading joblib model %s' % joblib_file)
            self._model = joblib.load(joblib_file)
        else:
            self._model = None
        self._preds = {}

    def predict(self, image: Image)-> dict:
        if not self._model:
            raise('Error! Model not assigned.')
        self._preds = self._model.predict(image)
        return self._preds

    def format(self, preds):
        return preds


class Pong(BaseModel):
    def predict(self):
        return {'Pong': 'Pong'}


class Vazios(BaseModel):
    def __init__(self):
        self._model = VazioModel()


class Peso(BaseModel):
    def __init__(self, linear=False):
        self._model = PesoModel(linear)


class Peso2(BaseModel):
    def __init__(self):
        self._model = PesoModel2()


class Naive(BaseModel):
    def __init__(self):
        self._model = NaiveModel()


class SSD(BaseModel):
    def __init__(self):
        self._model = SSDMobileModel()


class Encoder(BaseModel):
    def __init__(self):
        self._model = EncoderModel()


class VazioSVM(BaseModel):
    def __init__(self):
        self._model = VazioSVMModel()
