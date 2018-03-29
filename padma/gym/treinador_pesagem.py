"""Permite treinar o modelo definido em models/peso/peso.py.

Precisa de um csv no formato:
    _id,numero,peso,volume
    sendo _id o ObjectId da imagem no GridFS

Uma vez treinado, salva os histogramas e os pesos(em labels)
Para treinar novamente, basta excluir arquivos .pkl

"""
import csv
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
from bson.objectid import ObjectId
from gridfs import GridFS
from PIL import Image
from pymongo import MongoClient
from sklearn.metrics import explained_variance_score, mean_absolute_error, \
    r2_score
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from ajna_commons.flask.conf import (DATABASE, MONGODB_URI)
from ajna_commons.conf import ENCODE
from padma.models.peso.peso import PesoModel
from padma.models.bbox.bbox import NaiveModel
from padma.models.conteiner20e40.bbox import SSDMobileModel

pesomodel = PesoModel()
bboxmodel = NaiveModel()
bboxmodel = SSDMobileModel()

BASE_PATH = os.path.dirname(__file__)
HIST_FILE = os.path.join(BASE_PATH, 'histograms.npy')
LABEL_FILE = os.path.join(BASE_PATH, 'labels.npy')
CSV_FILE = os.path.join(BASE_PATH, 'pesovolexport.csv')
IMGOUT_PATH = os.path.join(BASE_PATH, 'images')


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
        numero_index = linha.index('numero')
        peso_index = linha.index('peso')
        volume_index = linha.index('volume')
        for ind, linha in enumerate(reader):
            if ind % 100 == 0 and ind != 0:
                print(ind)
                print(histograms[-1])
            print('.', end='', flush=True)
            img_file = os.path.join(IMGOUT_PATH,
                                    linha[numero_index] + '.jpg')
            im = None
            if os.path.exists(img_file):
                image = Image.open(img_file)
                im = np.asarray(image)
            else:
                grid_out = fs.get(ObjectId(linha[id_index]))
                tempfile = 'temp.jpg'
                with open(tempfile, 'wb') as temp:
                    temp.write(grid_out.read())
                image = Image.open(tempfile)
                # print(image.size)
                bboxes = bboxmodel.predict(image)
                if len(bboxes) > 0:
                    bbox = bboxes[0].get('bbox')
                    im = np.asarray(image)
                    im = im[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                    imageio.imwrite(img_file, im)
            if im is not None:
                with np.errstate(invalid='raise', divide='raise'):
                    try:
                        params = list(pesomodel.hist(im, n_bins=20))
                        params.extend([(im.shape[1] / im.shape[0])])
                        histograms.append(params)
                        labels.append((float(linha[peso_index]),
                                       float(linha[volume_index])))
                    except FloatingPointError:
                        print('Floating point error')
                        pass
        save_histograms(histograms, labels)
        labels = np.array(labels)
    return histograms, labels


def save_histograms(histograms, labels, sufix=''):
    with open(HIST_FILE+sufix, 'wb') as out:
        np.save(out, np.array(histograms))
    with open(LABEL_FILE+sufix, 'wb') as out:
        np.save(out, labels)


def load_histograms():
    print('Loading histograms...')
    if not os.path.exists(HIST_FILE):
        return None, None
    with open(HIST_FILE, 'rb') as pkl:
        histograms = np.load(pkl)
    with open(LABEL_FILE, 'rb') as pkl:
        labels = np.load(pkl)
    return histograms, labels


def train():
    pass


def evaluate(labels_test, labels_predicted,
             labels_train,
             labels_predicted_train):
    print()
    print('Variance score', explained_variance_score(
        labels_test, labels_predicted))

    print('R2 train', r2_score(labels_train,
                               labels_predicted_train))

    print('R2', r2_score(labels_test, labels_predicted))
    print('MAE', mean_absolute_error(labels_test, labels_predicted))
    print('Média', labels_train.mean())


def train_and_evaluate(histograms, labels):
    cont = len(histograms)
    print(cont)
    train = (cont // 5 + 1) * 4
    test = (cont // 5 + 1)
    print(train, test)
    histograms_train = histograms[:train]
    labels_train = labels[:train]
    histograms_test = histograms[-test:]
    labels_test = labels[-test:]
    pesomodel.train(histograms_train, labels_train)
    labels_predicted = pesomodel.model.predict(histograms_test)
    labels_predicted_train = pesomodel.model.predict(histograms_train)
    evaluate(labels_test, labels_predicted,
             labels_train,
             labels_predicted_train)
    return labels_predicted, labels_test


def refine(histograms, labels_predicted, labels_test):
    # Eliminar outliers
    new_histograms = []
    new_labels = []
    all_labels_predicted = pesomodel.model.predict(histograms)
    for peso, peso_pred, histo in zip(pesos, all_labels_predicted,
                                      histograms):
        razao = abs(peso - peso_pred) / peso
        if razao < .3:
            new_histograms.append(histo)
            new_labels.append(peso)
    return new_histograms, np.array(new_labels)


def train_and_refine(histograms, labels, prefix=''):
    print('Linear - TODOS')
    labels_predicted, labels_test = train_and_evaluate(histograms, labels)
    plt.scatter(labels_test, labels_predicted)
    plt.show()
    new_histograms, new_labels = refine(
        histograms, labels_predicted, labels_test)
    print('Linear - Sem outliers')
    print(len(new_histograms), len(new_labels))
    labels_predicted, labels_test = train_and_evaluate(
        new_histograms, new_labels)
    print()
    plt.scatter(labels_test, labels_predicted)
    plt.show()
    save_histograms(new_histograms, new_labels, prefix)
    return new_histograms, new_labels


if __name__ == '__main__':
    # histograms, labels = load_histograms()
    # TODO: Save labels for pesos and volumes
    #pesos = labels
    # if histograms is None:
    if True:
        histograms, labels = make_histograms()
        pesos = labels[:, 0]
        volumes = labels[:, 1]

    refined_histo, refined_label = train_and_refine(histograms, pesos)

    print('Quadratic - Sem outliers')
    quadratic = PolynomialFeatures(degree=2)
    histo_2 = quadratic.fit_transform(refined_histo)
    linear = LinearRegression()
    linear.fit(histo_2[528:], refined_label[528:])
    labels_predicted = linear.predict(histo_2[-132:])
    labels_test = refined_label[-132:]
    evaluate(labels_test, labels_predicted,
             labels_test, labels_predicted)
    plt.scatter(labels_test, labels_predicted)
    plt.show()


    print('RANSAC')
    ransac = RANSACRegressor(LinearRegression(), min_samples=100)
    ransac.fit(histograms, labels)
    labels_predicted = ransac.predict(histograms[-200:])
    labels_test = labels[-200:]
    evaluate(labels_test, labels_predicted,
             labels_test, labels_predicted)
    plt.scatter(labels_test, labels_predicted)
    plt.show()



    print('Random Forest')
    forest = RandomForestRegressor()
    forest.fit(histograms[:800], labels[:800])
    labels_predicted = forest.predict(histograms[-200:])
    labels_test = labels[-200:]
    evaluate(labels_test, labels_predicted,
             labels_test, labels_predicted)
    plt.scatter(labels_test, labels_predicted)
    plt.show()

    # train_and_refine(histograms, volumes, 'volume')

    # Print outliers
    """for ind, peso in enumerate(labels_predicted):
        if peso > 27000 or peso < 1000:
            print(ind, peso, labels_test[ind])
    """
