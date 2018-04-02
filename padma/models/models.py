# from keras.applications import ResNet50, imagenet_utils

from padma.models.bbox.bbox import NaiveModel
from padma.models.vazios.vazios import VazioModel
from padma.models.peso.peso import PesoModel
from padma.models.conteiner20e40.bbox import SSDMobileModel


class ModelBase():
    def __init__(self):
        self._model = None
        self._preds = {}

    def predict(self, data):
        if not self._model:
            raise('Error! Model not assigned.')
        self._preds = self._model.predict(data)
        return self._preds

    def format(self, preds):
        return preds


class Pong(ModelBase):
    def predict(self):
        return 'Pong'


class ResNet(ModelBase):
    def __init__(self, weights='imagenet'):
        pass
        # self._model = ResNet50(weights=weights)

    def format(self, preds):
        result_set = []
        # result_set = imagenet_utils.decode_predictions(preds)
        output = []
        for (_, label, prob) in result_set[0]:
            output.append({'label': label, 'probability': float(prob)})
        return output


class Vazios(ModelBase):
    def __init__(self):
        self._model = VazioModel()


class Peso(ModelBase):
    def __init__(self, linear=False):
        self._model = PesoModel(linear)


class Naive(ModelBase):
    def __init__(self):
        self._model = NaiveModel()


class SSD(ModelBase):
    def __init__(self):
        self._model = SSDMobileModel()
