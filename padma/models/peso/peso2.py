import os

import numpy as np
# from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib

N_BINS = 16
BASE_PATH = os.path.dirname(__file__)
HIST_FILE = os.path.join(BASE_PATH, 'histo.npy')
LABEL_FILE = os.path.join(BASE_PATH, 'label.npy')
MODEL_FILE = os.path.join(BASE_PATH, 'ForestPeso.pkl')


class PesoModel2():
    def __init__(self, bins=N_BINS, retrain=False):
        self._bins = bins
        self.model = joblib.load(MODEL_FILE)
        if retrain:
            with open(HIST_FILE, 'rb') as pkl:
                histograms = np.load(pkl)
            with open(LABEL_FILE, 'rb') as pkl:
                labels = np.load(pkl)
            self.model.fit(histograms, labels)

    def hist(self, img):
        histo = np.histogram(img, bins=self._bins)
        return histo[0][:N_BINS - 1]

    def predict(self, image):
        peso = self.model.predict([self.hist(image)])
        return [{'peso': peso[0]}]
