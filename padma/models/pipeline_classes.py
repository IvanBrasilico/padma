"""Classes que serão utilizadas em pipelines de tratamento.

Modelos dinâmicos precisam utilizar classes do python que estejam instaladas
no padma-venv, senão o pickle não conseguirá carregar.

Caso seja necessário classe personalizada no modelo dinâmico, não bastará
publicar o arquivo pickle, é necessário colocar o fonte da classe aqui.

"""
import numpy as np

NBINS = 16


class Histogram():
    def fit(self, _):
        return self

    def transform(self, img_array):
        return [np.histogram(img_array, bins=NBINS)[0].tolist()]
