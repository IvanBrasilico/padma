import os
from werkzeug.wsgi import DispatcherMiddleware

from padma.main import app

application = DispatcherMiddleware(app,
                                   {
                                       '/padma': app
                                   })
