import os
from werkzeug.wsgi import DispatcherMiddleware

from padma.app import app

application = DispatcherMiddleware(app,
                                   {
                                       '/padma': app
                                   })
