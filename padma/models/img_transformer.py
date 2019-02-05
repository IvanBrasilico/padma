import numpy as np
from sklearn.base import TransformerMixin
from PIL import Image
SIZE = 128, 128


class ImgTansformer(TransformerMixin):

    def transform(self, image, *_):
        image = image.resize(SIZE, Image.ANTIALIAS)
        image_array = np.asarray(image).astype('float32')
        image_array = image_array[:, :, 0] / 255
        return [np.reshape(image_array, (image_array.shape[0] * image_array.shape[1]))]

    def fit(self, *_):
        return self
