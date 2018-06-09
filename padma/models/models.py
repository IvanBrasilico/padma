from PIL import Image

from padma.models.bbox.bbox import NaiveModel
from padma.models.vazios.vazios import VazioModel
from padma.models.peso.peso import PesoModel
from padma.models.peso.peso2 import PesoModel2
from padma.models.conteiner20e40.bbox import SSDMobileModel


class ModelBase():
    def __init__(self):
        self._model = None
        self._preds = {}

    def predict(self, image: Image)-> dict:
        if not self._model:
            raise('Error! Model not assigned.')
        self._preds = self._model.predict(image)
        return self._preds

    def format(self, preds):
        return preds


class Pong(ModelBase):
    def predict(self):
        return {'Pong': 'Pong'}


class Vazios(ModelBase):
    def __init__(self):
        self._model = VazioModel()


class Peso(ModelBase):
    def __init__(self, linear=False):
        self._model = PesoModel(linear)

class Peso2(ModelBase):
    def __init__(self):
        self._model = PesoModel2()

class Naive(ModelBase):
    def __init__(self):
        self._model = NaiveModel()


class SSD(ModelBase):
    def __init__(self):
        self._model = SSDMobileModel()
