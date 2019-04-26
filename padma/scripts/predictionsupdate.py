"""Script de linha de comando para integração do Sistema PADMA.

ESTE Script difere do Script predictions_update por instanciar os
modelos dentro dele ao invés de consultar Servidor PADMA. Deve ser
utilizado apenas se houver necessidade de atualizar uma grande
quantidade de predições de forma rápida.

Script de linha de comando para fazer atualização 'manual'
dos metadados do módulo AJNA-PADMA nas imagens. Note-se que os metadados
de predições ficam em metadata.predictions, e este campo é uma lista de
lista de dicionários. Como pode haver mais de um contêiner na imagem, foi
necessário gravar uma lista neste campo, sendo item 0 para imagem 0 e
item 1 para imagem 1.

Importante: todos os modelos precisam atuar sobre um recorte da imagem
orginal, EXCETO os modelos treinados justamente para detectar este recorte.
Assim, serão filtrados apenas os registros que possuam a chave bbox para
recorte, a menos que o modelo selecionado seja um dos modelos próprios para
detecção do objeto contêiner (lista BBOX_MODELS do integracao.padma).

Args:

    modelo: modelo a consultar

    campo: campo das predições a atualizar

    tamanho: tamanho do lote de atualização/limite de registros da consulta

    batch_size: quantidade de consultas simultâneas, isto é, tamanho da matriz
    que será passada de uma só vez ao método predict do modelo

    update_date: se passado, a partir da data informada sobrescreverá a
     predição existente

"""
import datetime
import time

import click
import numpy as np
from ajna_commons.utils.images import generate_batch
from bson.objectid import ObjectId

from padma.db import mongodb as db
from padma.models.conteiner20e40.bbox import SSDMobileModel
from padma.models.encoder.encoder import EncoderModel
from padma.models.peso.peso2 import PesoModel2
from padma.models.vazios.vazio2 import VazioSVMModel

BBOX_MODELS = ['ssd']
PROJECTION = ['metadata.predictions']  # Economia de I/O


def monta_filtro(model: str, limit: int,
                 update: str) -> dict:
    """Retorna filtro para MongoDB."""
    filtro = {'metadata.contentType': 'image/jpeg'}
    # Modelo que cria uma caixa de coordenadas para recorte é pré requisito
    # para os outros modelos. Assim, outros modelos só podem rodar em registros
    # que já possuam o campo bbox (bbox: exists: True)
    if model not in BBOX_MODELS:
        filtro['metadata.predictions.bbox'] = {'$exists': True}
    if update is None:
        if model in BBOX_MODELS:
            filtro['metadata.predictions.bbox'] = {'$exists': False}
        else:
            filtro['metadata.predictions.' + model] = {'$exists': False}
    else:
        try:
            dt_inicio = datetime.datetime.strptime(update, '%d/%m/%Y')
        except ValueError:
            print('--update: Data em formato inválido!!!')
            return None
        print(dt_inicio)
        filtro['metadata.dataescaneamento'] = {'$gt': dt_inicio}

    return filtro


def mostra_tempo_final(s_inicial, registros_vazios, registros_processados):
    """Exibe mensagem de tempo decorrido."""
    s1 = time.time()
    elapsed = s1 - s_inicial
    horas = elapsed // 3600
    minutos = (elapsed % 3600) // 60
    segundos = elapsed % 60
    print('%d:%d:%d' % (horas, minutos, segundos),
          'registros vazios', registros_vazios,
          'registros processados', registros_processados)


BATCH_SIZE = 256
TAMANHO = 2560
MODEL = 'ssd'


@click.command()
@click.option('--modelo', help='Modelo de predição a ser consultado',
              required=True)
@click.option('--campo', help='Nome do campo a atualizar.'
                              + 'Se omitido, usa o nome do modelo.',
              default='')
@click.option('--limit',
              help='Tamanho do lote (padrão ' + str(TAMANHO) + ')',
              default=TAMANHO)
@click.option('--batch_size',
              help='Batch Size (consultas simultâneas - padrão ' +
                   str(BATCH_SIZE) + ')',
              default=BATCH_SIZE)
@click.option('--update_date', default=None,
              help='Reescrever dados existentes.'
                   + 'Passa por cima de dados existentes - especificar '
                   + 'data inicial (para não começar sempre do mesmo ponto)'
                   + ' no formato DD/MM/AAAA.')
def predictions_update(modelo, campo, limit, batch_size, update_date):
    """Consulta modelo e grava predições de retorno no MongoDB."""
    model = None
    if modelo == 'index':
        model = EncoderModel()
    elif modelo == 'peso':
        model = PesoModel2()
    elif modelo == 'ssd':
        model = SSDMobileModel()
    elif modelo == 'vazio':
        model = VazioSVMModel()
    if model is None:
        print('Modelo %s não implementado.' % modelo)
        return False
    if not campo:
        if modelo == 'ssd':
            campo = 'bbox'
        else:
            campo = modelo
    filtro = monta_filtro(campo, limit, update_date)
    if not filtro:
        return False
    encontrados = db['fs.files'].count_documents(filtro)
    print('Iniciando processamento de %s registros com o modelo %s, campo %s.'
          ' (disponiveis %s)'
          % (limit, modelo, campo, encontrados))
    batch_gen = generate_batch(db, filtro=filtro, projection=PROJECTION,
                               batch_size=batch_size, limit=limit,
                               recorta=modelo not in BBOX_MODELS)

    X = np.zeros((batch_size, *model.input_shape), dtype=np.float32)
    y = []
    original_images = []
    s = time.time()
    total = 0
    turns = 0
    for batch, rows in batch_gen:
        if len(batch) == 0:
            break
        for i, (images, row) in enumerate(zip(batch, rows)):
            original_images.append(images[0])
            image_array = model.prepara(images[0])
            s0 = time.time()
            if modelo in ['peso', 'vazio']:
                X[i, :] = image_array
            else:
                X[i, :, :, :] = image_array
        s1 = time.time()
        print('Montou X em %0.2f ' % (s1 - s0))
        if modelo == 'ssd':
            preds = model.predict_batch(X, original_images)
        else:
            preds = model.model.predict(X)
            if modelo == 'index':
                preds = preds.reshape(-1, 128).astype(np.float32)
        y.append(preds)
        s2 = time.time()
        print('Fez predição em %s. (batch size: %s)' % ((s2 - s1), batch_size))
        # print(indexes)
        if modelo in ['peso', 'vazio']:
            ystack = np.vstack(y).astype(np.float32).flatten()
        elif modelo == 'index':
            ystack = np.vstack(y).astype(np.float32)
        # print(y)
        if modelo == 'ssd':
            # Processa dicionário
            for i, bboxes in preds.items():
                index_row = rows[i]
                _id = index_row['_id']
                print(index_row)
                # print(bboxes)
                update_state = db.fs.files.update_one(
                    {'_id': ObjectId(_id)},
                    {'$set': {'metadata.predictions': bboxes}}
                )
                total = total + update_state.modified_count
                # novo = db.fs.files.find_one({'_id': ObjectId(_id)})
                # print(novo)
        else:
            for i in range(batch_size):
                if modelo == 'peso':
                    new_list = float(ystack[i])
                elif modelo == 'vazio':
                    new_list = float(ystack[i]) < 0.5
                elif modelo == 'index':
                    new_list = ystack[i, :].tolist()
                # print(new_list)
                index_row = rows[i]
                print(index_row)

                _id = index_row['_id']
                old_predictions = index_row['metadata']['predictions']
                # print(old_predictions)
                new_predictions = old_predictions
                new_predictions[0][campo] = new_list
                # print(new_predictions)
                update_state = db.fs.files.update_one(
                    {'_id': ObjectId(_id)},
                    {'$set': {'metadata.predictions': new_predictions}}
                )
                # novo = db.fs.files.find_one({'_id': ObjectId(_id)})
                # print(novo)
                total = total + update_state.modified_count
        s3 = time.time()
        print('Atualizou banco em %s' % (s3 - s2))
        turns += 1
        print('Total imagens %s. Batchs processados: %s'
              % (total, turns))
    s4 = time.time()
    elapsed = s4 - s
    tempo_imagem = 0 if (total == 0) else (elapsed / total)
    print('Tempo total: %s. Total imagens %s. Por imagem: %s'
          % (elapsed, total, tempo_imagem))


if __name__ == '__main__':
    s0 = time.time()
    predictions_update()
    s1 = time.time()
    print(
        'Tempo total de execução em segundos: {0:.2f}'.format(s1 - s0))
