import numpy as np
import os
from keras.models import load_model
from PIL import Image

BASE_PATH = os.path.dirname(__file__)
MODEL_FILE = os.path.join(BASE_PATH, 'encoder.h5')

SIZE = 128, 128
noise_factor = 0.4


class EncoderModel():
    def __init__(self):
        self.model = load_model(MODEL_FILE)

    def image_prepare(image):
        image = Image.open(image)
        image = image.resize(SIZE, Image.ANTIALIAS)
        image_array = np.asarray(image).astype('float32')
        image_array = image_array[:, :, 0] / 255
        image_array = np.expand_dims(image_array, axis=3)
        # image_noisy = image_array + noise_factor * \
        #    np.random.normal(loc=0.5, scale=0.33, size=image_array.shape)
        # image_noisy = np.clip(image_noisy, 0., 1.)
        return [image_array]  # , image_noisy

    def predict(self, image):
        code = self.model.predict(self.image_prepare(image))
        return [{'code': code[0]}]
