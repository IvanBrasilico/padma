import os

import numpy as np
from PIL import Image
from sklearn.externals import joblib

SIZE = 128, 128
BASE_PATH = os.path.dirname(__file__)
MODEL_FILE = os.path.join(BASE_PATH, 'SVCVazioDificeis.pkl')


class VazioSVMModel():
    def __init__(self, size=SIZE):
        self.size = size
        self.model = joblib.load(MODEL_FILE)

    def image_prepare(self, image: Image):
        image = image.resize(self.size, Image.ANTIALIAS)
        image_array = np.asarray(image).astype('float32')
        # del image
        image_array = image_array[:, :, 0] / 255
        print(image_array.shape)
        image_array = np.reshape(image_array,
                                 image_array.shape[0] * image_array.shape[1])
        return image_array

    def predict(self, image: Image):
        # O modelo SVM foi treinado em classificação binária
        # 0 para vazio e 1 para não vazio
        y = self.model.predict([self.image_prepare(image)]).tolist()
        vazio = [{'vazio': y_ == 0} for y_ in y]
        return vazio
