"""Permite treinar o modelo definido em models/peso/peso.py.

Precisa de um csv no formato:
    _id,numero,peso,volume
    sendo _id o ObjectId da imagem no GridFS

Uma vez treinado, salva os histogramas e os pesos(em labels)
Para treinar novamente, basta excluir arquivos .pkl

"""
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
from bson.objectid import ObjectId
from gridfs import GridFS
from PIL import Image
from pymongo import MongoClient
from sklearn import linear_model
from sklearn.metrics import explained_variance_score, mean_absolute_error, \
    mean_squared_log_error
from scipy import misc
from ajna_commons.flask.conf import (DATABASE, MONGODB_URI)
from ajna_commons.conf import ENCODE
from padma.models.peso.peso import PesoModel
from padma.models.bbox.bbox import NaiveModel

modelclass = PesoModel()
bboxclass = NaiveModel()

BASE_PATH = os.path.dirname(__file__)
HIST_FILE = os.path.join(BASE_PATH, 'histograms_peso.npy')
LABEL_FILE = os.path.join(BASE_PATH, 'labels_peso.npy')
CSV_FILE = os.path.join(BASE_PATH, 'pesovolexport.csv')


def make_histograms():
    histograms = []
    labels = []
    print('Connecting to MongoDB...')
    db = MongoClient(host=MONGODB_URI)[DATABASE]
    fs = GridFS(db)
    print('Making histograms...')
    with open(CSV_FILE, 'r', encoding=ENCODE, newline='') as csv_in:
        reader = csv.reader(csv_in)
        linha = next(reader)
        id_index = linha.index('id')
        peso_index = linha.index('peso')
        for ind, linha in enumerate(reader):
            if ind % 100 == 0:
                print(ind)
            print('.', end='', flush=True)
            grid_out = fs.get(ObjectId(linha[id_index]))
            tempfile = 'temp.jpg'
            with open(tempfile, 'wb') as temp:
                temp.write(grid_out.read())
            image = Image.open(tempfile)
            # print(image.size)
            bbox = bboxclass.predict(image).get('bbox')
            im = np.asarray(image)
            im = im[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            with np.errstate(invalid='raise', divide='raise'):
                try:
                    histograms.append(modelclass.hist(im))
                    labels.append(float(linha[peso_index]))
                except FloatingPointError:
                    print('Floating point error')
                    pass
        with open(HIST_FILE, 'wb') as out:
            np.save(out, np.array(histograms))
        with open(LABEL_FILE, 'wb') as out:
            labels = np.array(labels)
            np.save(out, labels)


def load_histograms():
    print('Loading histograms...')
    with open(HIST_FILE, 'rb') as pkl:
        histograms = np.load(pkl)
    with open(LABEL_FILE, 'rb') as pkl:
        labels = np.load(pkl)
    return histograms, labels


if __name__ == '__main__':
    if not os.path.exists(HIST_FILE):
        make_histograms()
    else:
        histograms, labels = load_histograms()
    # print(histograms)
    # print(labels)
    histograms_train = histograms[:800]
    labels_train = labels[:800]
    # print(histograms_train)
    # print(labels_train)
    histograms_test = histograms[-200:]
    labels_test = labels[-200:]
    modelclass.train(histograms_train, labels_train)
    labels_predicted = modelclass.model.predict(histograms_test)
    print()
    print('Variance', explained_variance_score(labels_test, labels_predicted))
    print('MAE', mean_absolute_error(labels_test, labels_predicted))
    # print('MSE', mean_squared_log_error(labels_test, labels_predicted))
    print('média dos pesos', labels.mean())
    plt.scatter(labels_test, labels_predicted)
    plt.show()
