import os
os.environ['DEBUG'] = '1'

from padma.modelserver import classify_process


classify_process()