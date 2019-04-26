from PIL import Image
import numpy as np
import os
import glob


def loadimages(path, input):
    numarquivos = int(len([name for name in os.listdir(path) if name.endswith('jpg')]))
    X = np.ndarray(shape=(numarquivos, input), dtype=np.float32)
    names = np.empty(shape=(numarquivos), dtype=np.object_)
    cont = 0
    for file in glob.glob(os.path.join(path,'*.jpg')):
        im = Image.open(file)
        X[cont] = np.asarray(im).reshape(input)
        #print(file)
        names[cont]=file
        cont +=1

    return X, names

        
def loadimagesnames(path, input):
    numarquivos = int(len([name for name in os.listdir(path) if name.endswith('jpg')]))
    names = np.empty(shape=(numarquivos), dtype=np.object_)
    cont = 0
    for file in glob.glob(os.path.join(path,'*.jpg')):
        names[cont]=file
        cont +=1

    return names
