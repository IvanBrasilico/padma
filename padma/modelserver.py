import json
import pickle
import time
import os
from threading import Thread

from ajna_commons.flask.conf import PADMA_REDIS, redisdb
from ajna_commons.flask.log import logger
from padma.models.conteiner20e40.bbox import SSDMobileModel
from padma.models.models import (BaseModel, Encoder, Naive, Peso, Peso2, Pong,
                                 Vazios, VazioSVM)


from padma.conf import BATCH_SIZE, MODEL_DIRECTORY, SERVER_SLEEP


def model_predict(model, _id, image):
    s0 = time.time()
    try:
        predictions = model.predict(image)
        # print('preds', output)
        output = {'success': True, 'predictions': predictions}
        dump = json.dumps(output)
        redisdb.set(_id, dump)
        s1 = time.time()
        print('Images classified in ', s1 - s0)
    except Exception as err:
        logger.debug('Erro em model_predict %s' % str(model))
        logger.debug(str(_id))
        logger.error(err, exc_info=True)
        output = {'success': False, 'predictions': [], 'erro': str(err)}
        dump = json.dumps(output)
        redisdb.set(_id, dump)

def load_models_hardcoded(modeldict):
    print('* Loading model PONG (ping-pong test if alive) *')
    modeldict['ping'] = Pong()
    print('* Loading model Vazios[vazio]...')
    modeldict['vazio'] = Vazios()
    print('* Model vazios loaded')
    print('* Loading model Peso Linear [pesol]...')
    modeldict['pesol'] = Peso(linear=True)
    print('* Model peso loaded')
    print('* Loading model Peso Random Forest [pesor] ...')
    modeldict['pesor'] = Peso()
    print('* Model peso random forest loaded')
    print('* Loading model Peso Random Forest 2 [peso]...')
    modeldict['peso'] = Peso2()
    print('* Model peso random forest 2 loaded')
    print('* Loading model Naive BBox[naive]...')
    modeldict['naive'] = Naive()
    print('* Model naive bbox loaded')
    print('* Loading model SSD BBox...')
    modeldict['ssd'] = SSDMobileModel()
    print('* Model SSD bbox loaded')
    print('* Loading model Indexador(index)...')
    modeldict['index'] = Encoder()
    print('* Model Index loaded')
    print('* Loading model Vazio SVM(vaziosvm)...')
    modeldict['vaziosvm'] = VazioSVM()
    print('* Model Vazio SVM loaded')



def model_process(model: str):
    local_model = BaseModel(os.path.join(MODEL_DIRECTORY, model))
    # continually poll for new images to classify
    while True:
        time.sleep(SERVER_SLEEP)
        queue = redisdb.lrange(PADMA_REDIS+model, 0, BATCH_SIZE - 1)
        if queue:
            model_key = 'nao definido'
            try:
                print('Processo 2 - processing image classify from queue')
                for cont, q in enumerate(queue, 1):
                    d = pickle.loads(q)
                    t = Thread(target=model_predict, args=(
                        [local_model, d['id'], d['image']]))
                    t.daemon = True
                    t.start()
            except Exception as err:
                logger.debug('Erro ao recuperar modelo %s' %
                             model_key)
                logger.debug(str(q))
                logger.error(err, exc_info=True)
                output = {'success': False, 'erro': str(err)}
                dump = json.dumps(output)
                redisdb.set(d['id'], dump)
            finally:
                redisdb.ltrim(PADMA_REDIS+model, cont, -1)


from multiprocessing import Process

def load_models_fromdisk(modeldict):
    try:
        models = os.listdir(MODEL_DIRECTORY)
        for model in models:
            p = Process(target=model_process, args=(model,))
            modeldict[model] = p
            p.start()
            # p.join()
    except FileNotFoundError:
        print('Caminho %s não encontrado!!!' % MODEL_DIRECTORY)


def classify_process():
    # Load the pre-trained models, only once.
    # Then wait for incoming queries on redis
    modeldict = dict()
    print('Carregando modelos Dinâmicos/processo')
    load_models_fromdisk(modeldict)
    print('Carregando modelos HARDCODED')
    load_models_hardcoded(modeldict)
    print('Fim dos carregamentos...')
    # continually poll for new images to classify
    while True:
        # attempt to grab a batch of images from the database
        time.sleep(SERVER_SLEEP)
        queue = redisdb.lrange(PADMA_REDIS, 0, BATCH_SIZE - 1)
        # loop over the queue
        if queue:
            cont = 0
            model_key = 'nao definido'
            try:
                print('Processing image classify from queue')
                for q in queue:
                    cont += 1
                    d = pickle.loads(q)
                    model_key = d.get('model')
                    model_item = modeldict.get(model_key)
                    if model_item is None:
                        logger.debug('Solicitado modelo não existente: "%s"' %
                                     model_key)
                        output = {'success': False,
                                  'erro': 'Modelo não existente: %s.' % model_key,
                                  'modelos':  list(modeldict.keys())
                                  }
                        dump = json.dumps(output)
                        redisdb.set(d['id'], dump)
                    else:
                        print('Vai para thread...')
                        logger.debug('Enviando para thread %s %s'
                                     % (model_key, model_item))
                        t = Thread(target=model_predict, args=(
                            [model_item, d['id'], d['image']]))
                        t.daemon = True
                        t.start()
            except Exception as err:
                logger.debug('Erro ao recuperar modelo %s' %
                             model_key)
                logger.debug(str(q))
                logger.error(err, exc_info=True)
                output = {'success': False, 'erro': str(err)}
                dump = json.dumps(output)
                redisdb.set(d['id'], dump)
            finally:
                redisdb.ltrim(PADMA_REDIS, cont, -1)


if __name__ == '__main__':
    classify_process()
