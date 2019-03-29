import json
import os
import pickle
import time
from threading import Thread

from ajna_commons.flask.conf import PADMA_REDIS, redisdb
from ajna_commons.flask.log import logger

from padma.conf import BATCH_SIZE, MODEL_DIRECTORY, SERVER_SLEEP
from padma.models.pipeline_classes import Histogram
from padma.models.conteiner20e40.bbox import SSDMobileModel
from padma.models.models import (BaseModel, Encoder, Naive, Peso2, Pong,
                                 VazioSVM)


class FailLoadModel:
    def __init__(self, error_msg):
        self.error_msg = error_msg

    def predict(self, ignore):
        return 'Erro ao carregar modelo: %s' % self.error_msg


def model_predict(model, _id, image):
    s0 = time.time()
    try:
        predictions = model.predict(image)
        # print('preds', output)
        output = {'success': True, 'predictions': predictions}
        dump = json.dumps(output)
        redisdb.set(_id, dump)
        s1 = time.time()
        logger.debug('Images classified in %s ' % (s1 - s0))
    except Exception as err:
        logger.error('Erro em model_predict %s' % str(model))
        logger.error(str(_id))
        logger.debug(err, exc_info=True)
        output = {'success': False, 'predictions': [], 'erro': str(err)}
        dump = json.dumps(output)
        redisdb.set(_id, dump)


def load_models_hardcoded(modeldict):
    logger.info('* Loading model PONG (ping-pong test if alive) *')
    modeldict['ping'] = Pong()
    # logger.info('* Loading model Vazios[vazio]...')
    # modeldict['vazio'] = Vazios()
    # logger.info('* Model vazios loaded')
    # logger.info('* Loading model Peso Linear [pesol]...')
    # modeldict['pesol'] = Peso(linear=True)
    # logger.info('* Model peso loaded')
    # logger.info('* Loading model Peso Random Forest [pesor] ...')
    # modeldict['pesor'] = Peso()
    # logger.info('* Model peso random forest loaded')
    logger.info('* Loading model Naive BBox[naive]...')
    modeldict['naive'] = Naive()
    logger.info('* Model naive bbox loaded')
    logger.info('* Loading model SSD BBox...')
    modeldict['ssd'] = SSDMobileModel()
    logger.info('* Model SSD bbox loaded')
    logger.info('* Loading model Indexador(index)...')
    modeldict['index'] = Encoder()
    logger.info('* Model Index loaded')
    logger.info('* Loading model Peso Random Forest 2 [peso]...')
    modeldict['peso'] = Peso2()
    logger.info('* Model peso random forest 2 loaded')
    logger.info('* Loading model Vazio SVM(vaziosvm)...')
    modeldict['vaziosvm'] = VazioSVM()
    logger.info('* Model Vazio SVM loaded')


def model_process(model: str):
    try:
        local_model = BaseModel(os.path.join(MODEL_DIRECTORY, model))
    except Exception as err:
        local_model = FailLoadModel(str(err))
        logger.debug(err, exc_info=True)
    # continually poll for new images to classify
    while True:
        time.sleep(SERVER_SLEEP)
        queue = redisdb.lrange(PADMA_REDIS, 0, BATCH_SIZE - 1)
        if queue:
            logger.debug(
                'Processo %s - processing image classify from queue' % model
            )
            for cont, q in enumerate(queue, 1):
                d = pickle.loads(q)
                model_key = d.get('model')
                logger.debug(model_key + ' - ' + model)
                if model_key == model:
                    try:
                        # t = Thread(target=model_predict, args=(
                        #    [local_model, d['id'], d['image']]))
                        # t.daemon = True
                        # t.start()
                        model_predict(local_model, d['id'], d['image'])
                    except Exception as err:
                        logger.error('Erro ao recuperar modelo %s' %
                                     model_key)
                        logger.error(str(q))
                        logger.debug(err, exc_info=True)
                        output = {'success': False, 'erro': str(err)}
                        dump = json.dumps(output)
                        redisdb.set(d['id'], dump)
                    finally:
                        redisdb.ltrim(PADMA_REDIS, cont, -1)


from multiprocessing import Process


def load_models_new_process(modeldict, models):
    """"""
    for model in models:
        p = Process(target=model_process, args=(model,))
        modeldict[model] = p
        p.start()
        # p.join()


def classify_process():
    # Load the pre-trained models, only once.
    # Then wait for incoming queries on redis
    modeldict = dict()
    logger.info('Carregando modelos Dinâmicos/processo')
    try:
        models = os.listdir(MODEL_DIRECTORY)
        load_models_new_process(modeldict, models)
    except FileNotFoundError:
        logger.warning('Caminho %s não encontrado!!!' % MODEL_DIRECTORY)

    logger.info('Carregando modelos HARDCODED')
    load_models_hardcoded(modeldict)
    logger.info('Fim dos carregamentos...')

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
                logger.debug('Processing image classify from queue')
                for q in queue:
                    cont += 1
                    d = pickle.loads(q)
                    model_key = d.get('model')
                    model_item = modeldict.get(model_key)
                    if model_item is None:
                        try:
                            model_key = str(model_key)
                        except TypeError as err:
                            logger.error(err, exc_info=True)
                            model_key = 'ERRO!!'
                        logger.debug('model_item None model_key %s' % model_key)
                        # Se existir mas não está carregado, carrega do disco.
                        if os.path.exists(os.path.join(MODEL_DIRECTORY, model_key)):
                            load_models_new_process(modeldict, [model_key])
                            output = {
                                'success': False,
                                'erro': 'Modelo %s ainda não carregado.' + \
                                        'Tente novamente.' % model_key
                            }
                        else:
                            logger.debug('Solicitado modelo não existente: "%s"' %
                                         model_key)
                            output = {'success': False,
                                      'erro': 'Modelo não existente: %s.' % model_key,
                                      'modelos': list(modeldict.keys())
                                      }
                        dump = json.dumps(output)
                        redisdb.set(d['id'], dump)
                    else:
                        # Testar se é modelo dinâmico. Se for, não faz nada
                        # pois há outro processo tratando.
                        if not isinstance(model_item, Process):
                            logger.debug('Enviando para thread %s %s'
                                         % (model_key, model_item))
                            t = Thread(target=model_predict, args=(
                                [model_item, d['id'], d['image']]))
                            t.daemon = True
                            t.start()
                            # model_predict(model_item, d['id'], d['image'])
            except Exception as err:
                logger.error('Erro ao recuperar modelo %s' %
                             model_key)
                logger.error(str(q))
                logger.debug(err, exc_info=True)
                output = {'success': False, 'erro': str(err)}
                dump = json.dumps(output)
                redisdb.set(d['id'], dump)
            finally:
                # Testar se é modelo dinâmico. Se for, não faz nada
                # pois há outro processo tratando.
                if not isinstance(model_item, Process):
                    redisdb.ltrim(PADMA_REDIS, cont, -1)


if __name__ == '__main__':
    classify_process()
