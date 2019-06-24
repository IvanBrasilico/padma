import os

import click
from ajna_commons.flask.conf import DATABASE, MONGODB_URI
from ajna_commons.utils.images import get_imagens_recortadas
from pymongo import MongoClient

ENCONTRADOS = {'metadata.carga.atracacao.escala': {'$ne': None},
               'metadata.contentType': 'image/jpeg'}


def get_cursor_filtrado(db,
                        vazio=False, limit=None):
    filtro = ENCONTRADOS
    filtro['metadata.carga.vazio'] = vazio
    filtro['metadata.predictions.vazio'] = vazio
    print(filtro)
    cursor = db['fs.files'].find(filtro)
    if limit:
        cursor.limit(limit)
    return cursor


def save_imagens(db,
                 path,
                 vazio=False,
                 limit=200):
    cursor = get_cursor_filtrado(db, vazio=vazio, limit=limit)
    try:
        os.mkdir(path)
    except FileExistsError:
        pass
    for linha in cursor:
        _id = linha['_id']
        for im in get_imagens_recortadas(db, _id):
            filename = str(_id) + '.jpg'
            im.save(os.path.join(path, filename))
            del im


@click.command()
@click.option('--limit',
              help='Tamanho do lote',
              default=200)
def exportaimagens(limit):
    db = MongoClient(host=MONGODB_URI)[DATABASE]
    save_imagens(db, 'vazio', vazio=True,
                 limit=limit)
    save_imagens(db, 'nvazio', vazio=False,
                 limit=limit)


if __name__ == '__main__':
    exportaimagens()
