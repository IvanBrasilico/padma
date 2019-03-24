from gridfs import GridFS
from pymongo import MongoClient
from ajna_commons.utils.images import get_imagens_recortadas

def get_cursor(db, filtro, projection=None, limit=None):
    if projection:
        cursor = db['fs.files'].find(filtro, projection)
    else:
        cursor = db['fs.files'].find(filtro)
    if limit:
        cursor = cursor[:limit]
    return cursor

def generate_batch(db, filtro, projection=None, batch_size=32, limit=None):
        """a generator for batches, so model.fit_generator can be used. """
        cursor = get_cursor(filtro, projection, limit)
        while True:
            images = []
            rows = []
            i = 0
            while i < batch_size:
                try:
                    row = next(cursor)
                except StopIteration:
                    break
                imgs = get_imagens_recortadas(db, row['_id'])
                images.append(imgs)
                rows.append(row)
                i += 1
            yield images, rows