import numpy as np
#X é sempre uma lista já codificada/comprimida pelo modelo

def euclidean_distance(lista, imagem):
    return distance(lista, imagem, True)

def chi_squared_distance(lista, imagem):
    return distance(lista, imagem, False)

def distance(lista, imagem, euclidean=True):
    squaredDistance = np.square(lista - imagem)
    if (euclidean==False):
        squaredDistance = squaredDistance / (np.abs(lista) + np.abs(imagem)+1)
    squareTree = np.sqrt(squaredDistance)
    distance = squareTree.sum(axis=1)
    return distance

def montaListaOrdenadaEuclidean(X, imagem):
    return montaListaOrdenada(X, imagem, True)
    
def montaListaOrdenadaChiSquare(X, imagem):
    return montaListaOrdenada(X, imagem, False)
    
def montaListaOrdenada(X, imagem, euclidean):
    sqrtDistance = distance(X, imagem, euclidean)
    order = np.argsort(sqrtDistance)
    return order

def listaBinaria(listSearch):
    listSearchMax = np.max(listSearch)
    listSearchMin = np.min(listSearch)
    listSearchRange = listSearchMax - listSearchMin
    listSearchNorm1a5 = np.ceil(((listSearch - listSearchMin) / listSearchRange) * 5)
    print(listSearchNorm1a5.max())
    print(listSearchNorm1a5.min())
    print(listSearchNorm1a5.shape)
    listSearchBinaryCode = np.zeros((listSearchNorm1a5.shape[0], listSearchNorm1a5.shape[1]*4), dtype=np.bool)
    cont1 = 0
    for linha in listSearchNorm1a5:
        cont2 = 0
        for elemento in linha:
            if elemento !=0:
              listSearchBinaryCode.itemset((cont1, cont2+int(elemento)-1), True)
            cont2+=4
        cont1+=1
    return listSearchBinaryCode

def montaListaOrdenadaHamming(listSearchBinaryCode, imagem):
    buscaBinaria = listaBinaria(imagem)
    hammingDistance = listSearchBinaryCode ^ buscaBinaria
    hammingDistance = hammingDistance.sum(axis=1)
    order = np.argsort(hammingDistance)
    return order

