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
import json
import time

import click
import numpy as np
from ajna_commons.utils.images import generate_batch
from bson.objectid import ObjectId

from padma.db import mongodb as db
from padma.models.encoder.encoder import SIZE, EncoderModel

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
    modelo = 'index'
    encoder = EncoderModel()
    if not campo:
        campo = modelo
    filtro = monta_filtro(campo, limit, update_date)
    if not filtro:
        return False
    encontrados = db['fs.files'].count_documents(filtro)
    print('Iniciando processamento de %s registros com o modelo %s, campo %s.'
          ' (disponiveis %s)'
          % (limit, modelo, campo, encontrados))
    batch_gen = generate_batch(db, filtro=filtro, projection=PROJECTION,
                               batch_size=batch_size, limit=limit)

    X = np.zeros((batch_size, *SIZE, 1), dtype=np.float32)
    y = []
    s = time.time()
    total = 0
    for batch, rows in batch_gen:
        if len(batch) == 0:
            break
        for i, (images, row) in enumerate(zip(batch, rows)):
            image_array = encoder.image_prepare(images[0])
            s0 = time.time()
            X[i, :, :, :] = image_array
        s1 = time.time()
        print('Montou X em %0.2f ' % (s1 - s0))
        indexes = encoder.model.predict(X)
        indexes = indexes.reshape(-1, 128).astype(np.float32)
        y.append(indexes)
        s2 = time.time()
        print('Fez predição em %s' % (s2 - s1))
        # print(indexes)
        ystack = np.vstack(y).astype(np.float32)
        for i in range(batch_size):
            new_list = ystack[i, :].tolist()
            index_row = rows[i]
            _id = index_row['_id']
            old_predictions = index_row['metadata']['predictions']
            new_predictions = old_predictions
            new_predictions[0]['index'] = new_list
            update_state = db.fs.files.update_one(
                {'_id': ObjectId(_id)},
                {'$set': {'metadata.predictions': json.dumps(new_predictions)}}
            )
            total = total + update_state.modified_count
    s3 = time.time()
    elapsed = s3 - s
    tempo_imagem = 0 if (total == 0) else (elapsed / total)
    print('Tempo total: %s. Total imagens %s. Por imagem: %s'
          % (elapsed, total, tempo_imagem))


if __name__ == '__main__':
    s0 = time.time()
    predictions_update()
    s1 = time.time()
    print(
        'Tempo total de execução em segundos: {0:.2f}'.format(s1 - s0))
