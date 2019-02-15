import io
import json
import numpy as np
import os
import pickle
import tempfile
import time
import uuid


from flask import (Flask, Response, flash, jsonify, redirect, render_template,
                   request, send_file, url_for)
from flask_bootstrap import Bootstrap
from flask_login import current_user, login_required
from flask_nav import Nav
from flask_nav.elements import Navbar, View
from flask_wtf.csrf import CSRFProtect
from PIL import Image

import ajna_commons.flask.login as login_ajna

from ajna_commons.flask.conf import (PADMA_REDIS,
                                     SECRET, redisdb)
from ajna_commons.flask.log import logger
from ajna_commons.utils.images import recorta_imagem

from padma.conf import CLIENT_SLEEP, CLIENT_TIMEOUT, MODEL_DIRECTORY


# initialize constants used for server queuing

tmpdir = tempfile.mkdtemp()


app = Flask(__name__, static_url_path='/static')
csrf = CSRFProtect(app)
Bootstrap(app)
nav = Nav()
nav.init_app(app)

def configure_app(mongodb):
    """Configurações gerais e de Banco de Dados da Aplicação."""
    app.config['DEBUG'] = os.environ.get('DEBUG', 'None') == '1'
    if app.config['DEBUG'] is True:
        app.jinja_env.auto_reload = True
        app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.secret_key = SECRET
    app.config['SECRET_KEY'] = SECRET
    app.config['SESSION_TYPE'] = 'filesystem'
    login_ajna.configure(app)
    login_ajna.DBUser.dbsession = mongodb
    app.config['mongodb'] = mongodb
    return app




def allowed_file(filename):
    """Check allowed extensions"""
    return '.' in filename and \
        filename.rsplit('.', 1)[-1].lower() in ['jpg']


@login_required
@app.route('/')
def index():
    if current_user.is_authenticated:
        return render_template('index.html')
    else:
        return redirect(url_for('commons.login'))


def call_model(model: str, image: Image)-> dict:
    """Grava requisição no redisdb e aguarda retorno até timeout.

        Args:
            model: string com uma chave do dicionário de modelos ativos
            image: PIL Image

        Returns:
            dict {'success', 'predictions', 'erro'}
            success: True ou False
            predictions: lista de dicts de predições em caso de sucesso
            erro: mensagem de erro se acontecer
    """
    logger.info('Enter Sandman - sending request to queue')
    # generate an ID then add the ID + image to the queue
    k = str(uuid.uuid4())
    d = {'model': model,
         'id': k,
         'image': image}
    modelname = PADMA_REDIS
    if '.pkl' in model:
        modelname += model
        print(modelname)
    redisdb.rpush(modelname, pickle.dumps(d, protocol=1))
    s0 = time.time()
    output = {'success': False, 'predictions': []}
    try:
        while True:
            # attempt to grab the output predictions
            output = redisdb.get(k)
            # check to see if our model has classified the input image
            if output is not None:
                output = output.decode('utf-8')
                output = json.loads(output)
                # delete the result from the database and exit loop
                redisdb.delete(k)
                break
            time.sleep(CLIENT_SLEEP)
            s1 = time.time()
            if s1 - s0 > CLIENT_TIMEOUT:  # Timeout
                logger.warning('Timeout!!!! Modelo %s ID %s' % (model, k))
                redisdb.delete(k)
                return {'success': False, 'erro': 'Timeout!!!'}
    finally:
        return output


@app.route('/predict', methods=['POST', 'GET'])
@csrf.exempt
# @login_required
def predict():
    # initialize the data dictionary that will be returned from the view
    data = [{'success': False}]
    s0 = None
    pil_image = None
    # ensure an image was properly uploaded to our endpoint
    model = request.args.get('model')
    image = request.files.get('image')
    if model:
        s0 = time.time()
        if image:
            pil_image = Image.open(io.BytesIO(image.read()))
        data = call_model(model, pil_image)
    # return the data dictionary as a JSON response
    if s0:
        s1 = time.time()
        logger.info('Results read from queue and returned in %f' % (s1 - s0))
    return jsonify(data)


@app.route('/image/<filename>')
@login_required
def image(filename):
    """Serializa a imagem do arquivo para stream HTTP."""
    filename = os.path.join(tmpdir, filename)
    image = open(filename, 'rb').read()
    return Response(response=image, mimetype='image/jpeg')


@app.route('/image_zoom/<filename>')
@login_required
def image_zoom(filename):
    """Recorta e serializa a imagem do arquivo para stream HTTP."""
    filename = os.path.join(tmpdir, filename)
    image = Image.open(filename)
    prediction = call_model('ssd', image)
    success = prediction.get('success')
    pred_bbox = prediction['predictions']
    if success:
        coords = pred_bbox[0]['bbox']
        img_io = recorta_imagem(image, coords)
    return send_file(img_io, mimetype='image/jpeg')


@app.route('/teste', methods=['GET', 'POST'])
@login_required
def teste():
    """Função simplificada para teste interativo de upload de imagem"""
    result = []
    ts = ''
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files.get('file')
        # if user does not select file, browser also
        # submit a empty part without filename
        """if file.filename == '':
            flash('No selected file')
            return redirect(request.url)"""
        if file:  # and allowed_file(file.filename):
            # os.remove('padma/static/temp.jpg')
            ts = str(time.time())
            filename = os.path.join(tmpdir, ts + '.jpg')
            with open(filename, 'wb') as temp:
                temp.write(file.read())
            # print('content', file.read())
            image = Image.open(filename)
            # success, pred_bbox = call_model('naive', image)
            prediction = call_model('ssd', image)
            success = prediction.get('success')
            pred_bbox = prediction['predictions']
            if success:
                result.append(json.dumps(image.size))
                result.append(json.dumps(pred_bbox))
                coords = pred_bbox[0]['bbox']
                im = np.asarray(image)
                im = im[coords[0]:coords[2], coords[1]:coords[3]]
                image = Image.fromarray(im)
                pred = call_model('vazio', image)
                result.append(json.dumps(pred))
                pred = call_model('vaziosvm', image)
                result.append(json.dumps(pred))
                pred = call_model('peso', im)
                result.append(json.dumps(pred))
                pred = call_model('pesor', im)
                result.append(json.dumps(pred))

    return render_template('teste.html', result=result, filename=ts + '.jpg')

@app.route('/modelos', methods=['GET', 'POST'])
# @login_required
def modelos():
    """Listar modelos disponíveis na tela, publicar novo modelo pipeline."""
    result = []
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            # os.remove('padma/static/temp.jpg')
            dest_filename = os.path.join(MODEL_DIRECTORY, file.filename)
            with open(dest_filename, 'wb') as model:
                model.write(file.read())
                # TODO: Open model
    result = call_model('Consulta', None)

    # return render_template('modelos.html', result=result)
    return jsonify(result)

@nav.navigation()
def mynavbar():
    items = [View('Home', 'index'),
             View('Testar modelos', 'teste'),
             View('Gerenciar modelos', 'modelos'),
             ]
    if current_user.is_authenticated:
        items.append(View('Sair', 'commons.logout'))
    return Navbar(*items)


nav.init_app(app)

app.config['DEBUG'] = os.environ.get('DEBUG', 'None') == '1'
if app.config['DEBUG'] is True:
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
app.secret_key = SECRET
app.config['SECRET_KEY'] = SECRET

if __name__ == '__main__':
    logger.info('* Starting web service...')
    app.config['DEBUG'] = True
    app.run(debug=app.config['DEBUG'], port=5002)
