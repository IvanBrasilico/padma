import json
import pickle
import time
import os
from sys import platform
from threading import Thread

from ajna_commons.flask.conf import redisdb
from ajna_commons.flask.log import logger
# from padma.models.conteiner20e40.bbox import SSDMobileModel
from padma.models.models import (BaseModel, Encoder, Naive, Peso, Peso2, Pong, Vazios,
                                 VazioSVM)

BATCH_SIZE = 10
SERVER_SLEEP = 0.10
MODEL_DIRECTORY = 'joblib_models'

def model_predict(model, _id, image):
    s0 = time.time()
    try:
        output = model.predict(image)
        # print('preds', output)
        dump = json.dumps(output)
        redisdb.set(_id, dump)
        s1 = time.time()
        print('Images classified in ', s1 - s0)
    except Exception as err:
        logger.debug('Erro em model_predict %s' % str(model))
        logger.debug(str(_id))
        logger.error(err, exc_info=True)
        output = {'success': False}
        dump = json.dumps(output)
        redisdb.set(_id, dump)

def load_models_hardcoded(modeldict):
    print('* Loading model PONG (ping-pong test if alive) *')
    modeldict['ping'] = Pong()
    print('* Loading model Vazios...')
    modeldict['vazio'] = Vazios()
    print('* Model vazios loaded')
    print('* Loading model Peso Linear (pesol)...')
    modeldict['pesol'] = Peso(linear=True)
    print('* Model peso loaded')
    print('* Loading model Peso Random Forest (pesor) ...')
    modeldict['pesor'] = Peso()
    print('* Model peso random forest loaded')
    print('* Loading model Peso Random Forest 2 (peso)...')
    modeldict['peso'] = Peso2()
    print('* Model peso random forest 2 loaded')
    print('* Loading model Naive BBox...')
    modeldict['naive'] = Naive()
    print('* Model naive bbox loaded')
    print('* Loading model SSD BBox...')
#     modeldict['ssd'] = SSDMobileModel()
    print('* Model SSD bbox loaded')
    print('* Loading model Indexador(index)...')
#    modeldict['index'] = Encoder()
    print('* Model Index loaded')
    print('* Loading model Vazio SVM(vaziosvm)...')
#    modeldict['vaziosvm'] = VazioSVM()
    print('* Model Vazio SVM loaded')



def load_models_fromdisk(modeldict):
    try:
        models = os.listdir(MODEL_DIRECTORY)
        for model in models:
            modeldict[model] = BaseModel(os.path.join(MODEL_DIRECTORY, model))
    except FileNotFoundError:
        print('Caminho %s não encontrado!!!' % MODEL_DIRECTORY)


def classify_process():
    # Load the pre-trained models, only once.
    # Then wait for incoming queries on redis
    modeldict = dict()
    load_models_hardcoded(modeldict)
    load_models_fromdisk(modeldict)
    if platform != 'win32':
        # continually poll for new images to classify
        while True:
            # attempt to grab a batch of images from the database
            for model_name, model in modeldict.items():
                queue = redisdb.lrange(model_name, 0, BATCH_SIZE - 1)
                # loop over the queue
                if queue:
                    try:
                        cont = 0
                        print('Processing image classify from queue')
                        for q in queue:
                            cont += 1
                            d = pickle.loads(q)
                            # TODO: Rodar model em Thread
                            t = Thread(target=model_predict, args=(
                                [model, d['id'], d['image']]))
                            t.daemon = True
                            t.start()
                    except Exception as err:
                        logger.debug('Erro ao recuperar modelo %s' %
                                     model_name)
                        logger.debug(str(q))
                        logger.error(err, exc_info=True)
                        output = {'success': False}
                        dump = json.dumps(output)
                        redisdb.set(d['id'], dump)
                    finally:
                        redisdb.ltrim(model_name, cont, -1)
            # sleep for a small amount
            time.sleep(SERVER_SLEEP)


if __name__ == '__main__':
    classify_process()
