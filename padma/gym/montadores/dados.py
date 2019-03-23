from gridfs import GridFS
from pymongo import MongoClient
from ajna_commons.flask.conf import DATABASE, MONGODB_URI
from ajna_commons.utils.images import get_imagens_recortadas


db = MongoClient(host=MONGODB_URI)[DATABASE]
fs = GridFS(db)

def get_cursor(filtro, limit=None):
    cursor = db['fs.files'].find(filtro)
    if limit:
        cursor = cursor[:limit]
    return cursor



def generate_batch(filtro, batch_size=32, limit=None):
        """a generator for batches, so model.fit_generator can be used. """
        cursor = get_cursor(filtro, limit)
        while True:
            images = []
            i = 0
            while i < batch_size:
                try:
                    row = next(cursor)
                except StopIteration:
                    break
                imgs = get_imagens_recortadas(db, row['_id'])
                images.append(imgs[0])
                i += 1
            yield images