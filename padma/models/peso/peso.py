import os

import numpy as np
from scipy import misc
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

N_BINS = 12
BASE_PATH = os.path.dirname(__file__)
HIST_FILE = os.path.join(BASE_PATH, 'histograms.npy')
LABEL_FILE = os.path.join(BASE_PATH, 'labels.npy')


class PesoModel():
    def __init__(self, bins=N_BINS, load=True, linear=False):
        self._bins = bins
        if linear:
            self.model = linear_model.LinearRegression()
        else:
            self.model = RandomForestRegressor()
        self.encoder = OneHotEncoder()
        self.encoder.fit([[i] for i in range(20)])
        if load:
            with open(HIST_FILE, 'rb') as pkl:
                histograms = np.load(pkl)
            with open(LABEL_FILE, 'rb') as pkl:
                labels = np.load(pkl)
            self.model.fit(histograms, labels)

    def hist(self, img, n_bins=None):
        if n_bins:
            self._bins = n_bins
        histo = np.histogram(img, bins=self._bins, density=True)
        return histo[0]

    def pesoimagem(self, file=None, image=None):
        if file:
            image = misc.imread(file)
        params = list(self.hist(image, n_bins=20))
        # TODO: get recinto_id
        recintoid = 1
        recinto = self.encoder.transform(
            [[recintoid]]
        ).toarray()
        params.extend(recinto[0])
        params.extend([(image.shape[1] / image.shape[0])])
        return self.model.predict([params])

    def train(self, histograms, labels):
        self.model.fit(histograms, labels)

    def predict(self, image):
        peso = self.pesoimagem(image=image)
        return [{'peso': peso[0]}]
