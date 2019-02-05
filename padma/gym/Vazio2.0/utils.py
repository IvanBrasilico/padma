import os

import numpy as np
from PIL import Image

def walk(dir_name):
    names = []
    for dirpath, dirnames, filenames in os.walk(dir_name):
        for filename in filenames:
            names.append(os.path.join(dirpath, filename))
    return names


SIZE = 128, 128

def image_array(filenames):
    result = []
    for filename in filenames:
        image = Image.open(filename)
        image = image.resize(SIZE, Image.ANTIALIAS)
        image_array = np.asarray(image).astype('float32')
        image_array = image_array[:, :, 0] / 255
        del image
        result.append(image_array)
    return result


def img_to_array(image):
    image = image.resize(SIZE, Image.ANTIALIAS)
    image_array = np.asarray(image).astype('float32')
    image_array = image_array[:, :, 0] / 255
    return [np.reshape(image_array, (image_array.shape[0] * image_array.shape[1]))]
    