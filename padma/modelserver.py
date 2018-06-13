import json
import pickle
import time
from threading import Thread
from sys import platform

from ajna_commons.flask.conf import (redisdb)
from ajna_commons.flask.log import logger

from padma.models.models import (Encoder, Naive, Peso, Peso2, Pong, Vazios)
from padma.models.conteiner20e40.bbox import SSDMobileModel

BATCH_SIZE = 10
SERVER_SLEEP = 0.10


def model_predict(model, _id, image):
    s0 = time.time()
    output = model.predict(image)
    print('preds', output)
    dump = json.dumps(output)
    redisdb.set(_id, dump)
    s1 = time.time()
    print('Images classified in ', s1 - s0)


def classify_process():
    # Load the pre-trained models, only once.
    # Then wait for incoming queries on redis
    modeldict = dict()
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
    modeldict['ssd'] = SSDMobileModel()
    print('* Model SSD bbox loaded')
    print('* Loading model Indexador(index)...')
    modeldict['index'] = Encoder()
    print('* Model Index loaded')

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
                    except TypeError as err:
                        logger.debug('Erro ao recuperar modelo %s' %
                                     model_name)
                        logger.debug(str(q))
                        logger.debug(err, exc_info=True)
                    finally:
                        redisdb.ltrim(model_name, cont, -1)
            # sleep for a small amount
            time.sleep(SERVER_SLEEP)
