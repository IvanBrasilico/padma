import numpy as np
import os
from PIL import Image
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin

from padma.models.bbox.bbox import NaiveModel
from padma.models.conteiner20e40.bbox import SSDMobileModel
from padma.models.encoder.encoder import EncoderModel
from padma.models.peso.peso import PesoModel
from padma.models.peso.peso2 import PesoModel2
from padma.models.vazios.vazio2 import VazioSVMModel
from padma.models.vazios.vazios import VazioModel
from padma.models.img_transformer import ImgTansformer

class BaseModel():
    def __init__(self, joblib_file=None):
        self._joblib = False
        if (joblib_file is not None) and os.path.exists(joblib_file):
            print('*Loading joblib model %s' % joblib_file)
            self._model = joblib.load(joblib_file)
            self._joblib = True
        else:
            self._model = None
        self._preds = {}

    def predict(self, image: Image) -> dict:
        if not self._model:
            raise ('Error! Model not assigned.')
        if self._joblib:
            y = self._model.predict(image)
            # print(y)
            # TODO: Houve um erro de design e os modelos estão acoplados...
            # Ao invés de retornar a predição diretamente, fazem uma transcrição/tradução
            # Corrigir isso, retornando a predição original padrão em todos os modelos
            self._preds = y.tolist() # [{'vazio': bool(y_ == 0.)} for y_ in y]
        else:
            self._preds = self._model.predict(image)
        # print(self._preds)
        return self._preds

    def format(self, preds):
        return preds


class Pong(BaseModel):
    def predict(self):
        return {'Pong': 'Pong'}


class Vazios(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
        self._model = VazioModel()


class Peso(BaseModel):
    def __init__(self, linear=False):
        BaseModel.__init__(self)
        self._model = PesoModel(linear)


class Peso2(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
        self._model = PesoModel2()


class Naive(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
        self._model = NaiveModel()


class SSD(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
        self._model = SSDMobileModel()


class Encoder(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
        self._model = EncoderModel()


class VazioSVM(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
        self._model = VazioSVMModel()
