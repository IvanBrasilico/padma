import os
from collections import Counter

import click
from ajna_commons.flask.conf import DATABASE, MONGODB_URI
from ajna_commons.utils.images import get_imagens_recortadas
from pymongo import MongoClient

NCMUNICO = {'metadata.contentType': 'image/jpeg',
            'metadata.carga.ncm': {'$size': 1},
            'metadata.carga.container.indicadorusoparcial': {'$ne': 's'}
            }


def get_cursor_filtrado(db, filtro, limit=None):
    print(filtro)
    cursor = db['fs.files'].find(filtro)
    if limit:
        cursor.limit(limit)
    return cursor


def save_imagens_ncm_unico(db,
                           path,
                           limit=None,
                           limitportipo=100):
    filtro = NCMUNICO
    cursor = get_cursor_filtrado(db, filtro, limit=limit)
    tipo_counter = Counter()
    try:
        os.mkdir(path)
    except FileExistsError:
        pass
    for linha in cursor:
        _id = linha['_id']
        ncms = linha.get('metadata').get('carga').get('ncm')
        ncms_encontrados = set()
        for ncm in ncms:
            posicao = ncm.get('ncm')[:4]
            ncms_encontrados.add(posicao)
        if len(ncms_encontrados) == 1:  # Achou 1 e somente 1 posição ncm
            posicao = list(ncms_encontrados)[0]
            if tipo_counter[posicao] < limitportipo:
                for im in get_imagens_recortadas(db, _id):
                    sub_path = os.path.join(path, posicao)
                    try:
                        os.mkdir(sub_path)
                    except FileExistsError:
                        pass
                    filename = str(_id) + '.jpg'
                    im.save(os.path.join(sub_path, filename))
                    del im
                    tipo_counter[posicao] += 1


@click.command()
@click.option('--limit',
              help='Tamanho do lote',
              default=1000)
@click.option('--limitportipo',
              help='Limite por NCM ',
              default=20)
def exportaimagens(limit, limitportipo):
    db = MongoClient(host=MONGODB_URI)[DATABASE]
    save_imagens_ncm_unico(db, 'ncmsunicos',
                           limit=limit, limitportipo=limitportipo)


if __name__ == '__main__':
    exportaimagens()
