import csv
import numpy as np
import os
import pickle
from bson.objectid import ObjectId
from gridfs import GridFS
from PIL import Image
from pymongo import MongoClient
from sklearn import linear_model
from scipy import misc
from ajna_commons.flask.conf import (DATABASE, MONGODB_URI)
from padma.models.peso.peso import PesoModel

modelclass = PesoModel()

BASE_PATH = os.path.dirname(__file__)
HIST_FILE = os.path.join(BASE_PATH, 'histograms_peso.pkl')
LABEL_FILE = os.path.join(BASE_PATH, 'labels_peso.pkl')
CSV_FILE = os.path.join(BASE_PATH, 'pesovolexport.csv')
histograms = []
labels = []


def make_histograms():
    print('Connecting to MongoDB...')
    db = MongoClient(host=MONGODB_URI)[DATABASE]
    fs = GridFS(db)
    print('Making histograms...')
    with open(CSV_FILE, 'r') as csv_in:
        reader = csv.reader(csv_in)
        linha = next(reader)
        id_index = linha.index('id')
        peso_index = linha.index('peso')
        for ind, linha in enumerate(reader):
            print('.', end='')
            if ind // 100 == 0:
                print(ind)
            grid_out = fs.get(ObjectId(linha[id_index]))
            tempfile = '/tmp/temp.jpg'
            with open(tempfile, 'wb') as temp:
                temp.write(grid_out.read())
            image = Image.open(tempfile)
            im = np.asarray(image)
            histograms.append(modelclass.hist(im))
            labels.append(linha[peso_index])
        with open(HIST_FILE, 'wb') as out:
            pickle.dump(histograms, out)
        with open(LABEL_FILE, 'wb') as out:
            pickle.dump(labels, out)


def load_histograms():
    print('Loading histograms...')
    with open(HIST_FILE, 'rb') as pkl:
        histograms = pickle.load(pkl)
    with open(LABEL_FILE, 'rb') as pkl:
        labels = pickle.load(pkl)


if __name__ == '__main__':
    if not os.path.exists(HIST_FILE):
        make_histograms()
    else:
        load_histograms()
