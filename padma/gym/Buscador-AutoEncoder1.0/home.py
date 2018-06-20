import pickle
import numpy as np
import os
from indexfunctions import *
from filefunctions import *

size = (256, 120)
inputsize = int(size[0]*size[1])
home = '/home/ivan/Área de Trabalho/ajna/amostras/stamps/'
dest = os.path.join(home, 'resize/')

with open('compressed.pkl', 'rb') as input:
        compressed = pickle.load(input)
print("Listas montadas")

if (plano):
    import tflearn
    from modelfully import *
    model, encoder, decoder = modelfully1(inputsize)
    model.load('plano/conteineresencoder.tflearn')
    encoding_model = tflearn.DNN(encoder, session=model.session)
    print("Modelo carregado")        
    listSearch = np.array(compressed, dtype=np.float32)
else:
    import tensorflow as tf

    
#listSearchBinaryCode = listaBinaria(listSearch)


#X, names = loadimages(dest, inputsize)
#i = 0
#imagem = X[i] # Aqui, na prática, a aplicação irá receber uma imagem e a transformar para procura
names = loadimagesnames(dest, inputsize)
print("Imagens carregadas")

def imagelist(i):
    imagem = X[i]
    order = montaListaOrdenadaEuclidean(listSearch, imagem, encoding_model)
    return names[order]

def imagelist2(imagem):
    order = montaListaOrdenadaEuclidean(listSearch, imagem, encoding_model)
    return names[order]

#print(imagelist())
#print(order)
