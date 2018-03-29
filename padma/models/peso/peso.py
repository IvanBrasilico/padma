import pickle
import numpy as np
from sklearn import linear_model
from scipy import misc
import os

n_bins = 32


class PesoModel():
    def __init__(self, bins=16, load=True):
        self._bins = bins
        if load:
            histograms = pickle.load(
                open(os.path.join(os.path.dirname(__file__), '..', 'vazios',
                                  'histograms.pkl'), 'rb'))
            labels = pickle.load(
                open(os.path.join(os.path.dirname(__file__), '..', 'vazios',
                                  'labels.pkl'), 'rb'))
            self.model = linear_model.LinearRegression()
            self.model.fit(histograms, labels)

    def hist(self, img):
        histo = np.histogram(img, bins=n_bins, density=True)
        return histo[0]

    def pesoimagem(self, file=None, image=None):
        if file:
            image = misc.imread(file)
        return self.model.predict([self.hist(image)])

    def train(self, histograms, labels):
        self.model.fit(histograms, labels)

    def predict(self, image):
        peso = self.pesoimagem(image=image)
        print('peso', peso[0])
        return {'peso': peso[0]}
