import os

import numpy as np
import tensorflow as tf
from keras.models import Model, load_model
from PIL import Image

BASE_PATH = os.path.dirname(__file__)
MODEL_FILE = os.path.join(BASE_PATH, 'autoencoderGPUNormal100Epoch.h5')

SIZE = 128, 128
noise_factor = 0.4


class EncoderModel():
    def __init__(self):
        graph = tf.get_default_graph()
        self.graph = graph
        with graph.as_default():
            autoencoder = load_model(MODEL_FILE)
            model = Model(inputs=autoencoder.input,
                          outputs=autoencoder.get_layer('encoder').output)
        self.model = model

    def image_prepare(self, image: Image):
        image = image.resize(SIZE, Image.ANTIALIAS)
        image_array = np.asarray(image).astype('float32')
        image_array = image_array[:, :, 0] / 255
        image_array = np.expand_dims([image_array], axis=3)
        return image_array

    def predict(self, image: Image):
        graph = self.graph
        with graph.as_default():
            model = self.model
            image_array = self.image_prepare(image)
            code = model.predict(image_array)
        return [{'code': code.reshape(128).tolist()}]
