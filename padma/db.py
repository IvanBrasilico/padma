"""Configura a conexão com Banco de Dados."""
from pymongo import MongoClient

from ajna_commons.flask.conf import DATABASE, MONGODB_URI

conn = MongoClient(host=MONGODB_URI)
mongodb = conn[DATABASE]
