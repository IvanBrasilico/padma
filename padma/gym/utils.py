import csv
import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
from PIL import Image
from pymongo import MongoClient

from sklearn.metrics import mean_squared_error, r2_score, \
    mean_absolute_error, median_absolute_error 


from ajna_commons.flask.conf import DATABASE, MONGODB_URI
from ajna_commons.utils.images import get_imagens_recortadas

sys.path.insert(0, '../../virasana/')
# from virasana.integracao import carga, CHAVES_GRIDFS
# from virasana.exportacao.utils import campos_mongo_para_lista

ENCONTRADOS = {'metadata.carga.atracacao.escala': {'$ne': None},
               'metadata.contentType': 'image/jpeg'}

def get_lista(db, start, end, vazio=None, max_linhas=None):
    filtro = ENCONTRADOS
    filtro['metadata.predictions.bbox'] = {'$exists': True}
    filtro['metadata.dataescaneamento'] = {
        '$gt': datetime.strptime(start, '%Y-%m-%d'),
        '$lt': datetime.strptime(end, '%Y-%m-%d')
    }
    if isinstance(vazio, bool):
        filtro['metadata.carga.vazio'] = vazio
    print(filtro)
    cursor = db['fs.files'].find(filtro)
    if max_linhas:
        cursor.limit(max_linhas)
    return cursor


def get_images(db, lista):
    imagens = []
    for linha in lista:
        _id = linha['_id']
        imagens.append([im for im in get_imagens_recortadas(db, _id)])
    return imagens

def monta_df(bins, inicio=None, fim=None, from_dir=None, max_rows=1000, vazio=False):
    """Conecta ao MongoDB, monta um dataframe com histograma e pesos.
    
    Conecta ao MongoDB, consulta imagens, recorta imagens,
    calcula histogramas, e monta um dataframe com bins do 
    histograma e pesos declarados.
    
    Args:
        bins: número de bins do histograma
        inicio, fim: datas a filtrar no Banco ('YYYY-mm-dd')
        from_dir: se fornecido, carrega de diretório ao invés do Banco de Dados
       
    Returns:
        pandas dataframe, lista de imagens
    """
    if from_dir:
        with open(os.path.join(from_dir, 'img_data.csv')) as csv_file:
            reader = csv.reader(csv_file)
            linha = next(reader)
            index_id = linha.index('_id')
            index_tara = linha.index('taracontainer')
            index_peso = linha.index('pesobrutoitem')
            pesos = []
            images = []

            for index, linha in enumerate(reader):
                try:
                    tara = float(linha[index_tara].replace(',', '.'))
                    peso = float(linha[index_peso].replace(',', '.'))
                    image_name = linha[index_id] + '_0'
                    pesos.append(tara + peso)
                    imagem = Image.open(os.path.join(from_dir, image_name))
                    images.append([np.asarray(imagem)])
                    del imagem
                    if index >= max_rows:
                        break
                except ValueError:
                    pass

    else:
        db = MongoClient(host=MONGODB_URI)[DATABASE]
        cursor = get_lista(db, inicio, fim, vazio)
        lista = [linha for linha in cursor]
        images = get_images(db, lista)
        pesos = []
        for linha in lista:
            carga = linha.get('metadata').get('carga')
            if carga:
                vazio = carga.get('vazio')
                container = carga.get('container')
                if container:
                    if vazio is False:
                        tara = float(container[0].get('taracontainer').replace(',', '.'))
                        peso = float(container[0].get('pesobrutoitem', '0').replace(',', '.'))
                        if container[0].get('indicadorusoparcial') == 's':
                            for cont in container[1:]:
                                peso += float(container[0].get(
                                             'pesobrutoitem', '0').replace(',', '.'))
                    else:
                        tara = float(container[0].get('tara(kg)').replace(',', '.'))
                        peso = 0
                    pesos.append(tara + peso)
    histograms = [np.histogram(np.asarray(image[0]), bins=bins)[0] for image in images]
    df = pd.DataFrame(histograms)
    df.columns = np.histogram(np.asarray(images[0][0]), bins=16)[1][1:]
    if len(pesos) == len(df):
        df['peso'] = pesos
    else:
        print('''Pesos não puderam ser recuperados.
              Se não for uma extração de vazios, deve haver erro na base
              ''')
        print('Número de pesos: %d. Número de imagens: %d' % 
              (len(pesos), len(df)))
    return df, images

def reg_plot(model, X_test, y_test):
    # Make predictions using the testing set
    y_pred = model.predict(X_test)

    # The coefficients
    # print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(y_test, y_pred))
    print("Mean absolute error: %.2f"
          % mean_absolute_error(y_test, y_pred))
    print("Median absolute error: %.2f"
          % median_absolute_error(y_test, y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_test, y_pred))

    # Plot outputs
    plt.scatter(y_test, y_test,  color='black')
    plt.scatter(y_test, y_pred, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())
    plt.show()
    
   

def df_plot(df, max_images):
    bins = len(df.columns) - 1

    df.hist()

    columns = 4
    rows = bins // 4
    fig=plt.figure(figsize=(12, 3 * rows))
    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        x = df[df.columns[i-1]]
        y = df['peso']
        plt.scatter(x, y)
    plt.show()

    for i in range(bins):
        print(i, np.corrcoef(df[df.columns[i]], df['peso']))


def monta_lista_ids_e_imagens(inicio, fim, max_linhas=100):
    db = MongoClient(host=MONGODB_URI)[DATABASE]
    cursor = get_lista(db, inicio, fim, max_linhas)
    lista = [linha for linha in cursor]
    images = get_images(db, lista)
    _ids = [linha['_id'] for linha in lista]
    return _ids, images

        
def monta_lista_ids_e_imagens(inicio, fim, max_linhas=100):
    db = MongoClient(host=MONGODB_URI)[DATABASE]
    cursor = get_lista(db, inicio, fim, max_linhas)
    lista = [linha for linha in cursor]
    images = get_images(db, lista)
    _ids = [linha['_id'] for linha in lista]
    return _ids, images

def view_imagens(imagens):
    plt.gray()
    fig=plt.figure(figsize=(16, 20))
    columns = 4
    rows = 4
    for i in range(1, columns*rows +1):
        img = imagens[i - 1]
        ax = fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()


