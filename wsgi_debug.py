import os
from werkzeug.serving import run_simple
from werkzeug.wsgi import DispatcherMiddleware

from ajna_commons.flask.conf import PADMA_URL

os.environ['DEBUG'] = '1'
from padma.main import app

if __name__ == '__main__':
    port = 5002
    if PADMA_URL:
        port = int(PADMA_URL.split(':')[-1])
    application = DispatcherMiddleware(app,
                                    {
                                        '/padma': app
                                    })
    run_simple('localhost', port, application, use_reloader=True)
