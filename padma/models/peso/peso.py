import pickle
import numpy as np
from sklearn import linear_model
from scipy import misc
import os

N_BINS = 32
BASE_PATH = os.path.dirname(__file__)
HIST_FILE = os.path.join(BASE_PATH, 'histograms_peso.npy')
LABEL_FILE = os.path.join(BASE_PATH, 'labels_peso.npy')


class PesoModel():
    def __init__(self, bins=N_BINS, load=True):
        self._bins = bins
        if load:
            with open(HIST_FILE, 'rb') as pkl:
                histograms = np.load(pkl)
            with open(LABEL_FILE, 'rb') as pkl:
                labels = np.load(pkl)
            self.model = linear_model.LinearRegression()
            self.model.fit(histograms, labels)

    def hist(self, img):
        histo = np.histogram(img, bins=self._bins, density=True)
        return histo[0]

    def pesoimagem(self, file=None, image=None):
        if file:
            image = misc.imread(file)
        return self.model.predict([self.hist(image)])

    def train(self, histograms, labels):
        self.model.fit(histograms, labels)

    def predict(self, image):
        peso = self.pesoimagem(image=image)
        return {'peso': peso[0]}
