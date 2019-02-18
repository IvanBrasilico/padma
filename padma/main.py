"""Módulo de entrada da aplicação web.

Interface web que faz proxy para a chamada de modelos.

Esta aplicação faz a autenticação dos clientes e um "proxy" para a chamada
dos modelos de aprendizado de máquina. Os modelos são servidos efetivamente
por outro processo "modelserver.py". A comunicação entre os dois processos
se dá via REDIS.

São responsabilidades desta aplicação:

    - Autenticação
    - Segurança
    - Tratamento de erros
    - Receber uma imagem, repassar para modelserver, aguardar resposta,\
formatar resposta e enviar para cliente. Controlar e avisar de timeout.

"""

import ajna_commons.flask.log as log
from ajna_commons.flask.flask_log import configure_applog
from padma.db import mongodb
from padma.views import configure_app

app = configure_app(mongodb)
configure_applog(app)
log.logger.info('Servidor (re)iniciado!')

if __name__ == '__main__':  # pragma: no cover
    app.run(debug=app.config['DEBUG'])
