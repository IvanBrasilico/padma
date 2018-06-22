# JSON server
# Args
#  model, params
# Returns
#   JSON dict with model response
import io
import json
import os
import pickle
import tempfile
import time
import uuid
# from base64 import b64encode
# from base64 import decodebytes
from sys import platform

import numpy as np
from flask import (Flask, Response, flash, jsonify, redirect, render_template,
                   request, send_file, url_for)
from flask_bootstrap import Bootstrap
from flask_login import current_user, login_required
from flask_nav import Nav
from flask_nav.elements import Navbar, View
from flask_wtf.csrf import CSRFProtect
# import redis
from PIL import Image
from pymongo import MongoClient

import ajna_commons.flask.login as login
from ajna_commons.flask.conf import DATABASE, MONGODB_URI, SECRET, redisdb

# from ajna_commons.flask.log import logger


PADMA_REDIS_KEY = 'PADMAKEY01'

# initialize constants used to control image spatial dimensions and
# data type
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANS = 3
IMAGE_DTYPE = 'float32'

# initialize constants used for server queuing
CLIENT_SLEEP = 0.10  # segundos
CLIENT_TIMEOUT = 10  # segundos
tmpdir = tempfile.mkdtemp()

# Configure app and DB Connection
db = MongoClient(host=MONGODB_URI)[DATABASE]
app = Flask(__name__, static_url_path='/static')
csrf = CSRFProtect(app)
Bootstrap(app)
nav = Nav()
login.login_manager.init_app(app)
login.configure(app)
login.DBUser.dbsession = db


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
        return redirect(url_for('login'))


def win32_call_model(model, image):
    """Síncrono, sem threads, para uso no desktop Windows."""
    from modelserver import model_dict
    model = model_dict[model]
    output = model.predict(image)
    return True, output


def call_model(model: str, image: Image):
    """Grava requisição no redisdb e aguarda retorno até timeout.

        Args:
            model: string com uma chave do dicionário de modelos ativos
            image: PIL Image

        Returns:
            True, dict com predições em caso de sucesso
            False, dict vazio em caso de timeout
    """
    if platform == 'win32':
        return win32_call_model(model, image)
    print('Enter Sandman - sending request to queue')
    # generate an ID then add the ID + image to the queue
    k = str(uuid.uuid4())
    d = {'id': k,
         'image': image}
    redisdb.rpush(model, pickle.dumps(d, protocol=1))
    s0 = time.time()
    while True:
        # attempt to grab the output predictions
        output = redisdb.get(k)
        # check to see if our model has classified the input image
        if output is not None:
            output = output.decode('utf-8')
            predictions = json.loads(output)
            # delete the result from the database and exit loop
            redisdb.delete(k)
            break
        time.sleep(CLIENT_SLEEP)
        s1 = time.time()
        if s1 - s0 > CLIENT_TIMEOUT:  # Timeout
            print('Timeout!!!!')
            redisdb.delete(k)
            return False, {}
    return True, predictions


def preprocess_image(image, prepare):
    # read the image in PIL format and prepare it for classification
    image = Image.open(io.BytesIO(image))
    if prepare:
        image = prepare(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        # ensure our NumPy array is C-contiguous as well,
        # otherwise we won't be able to serialize it
        image = image.copy(order='C')
    return image


@csrf.exempt
@app.route('/predict', methods=['POST'])
def predict():
    # preprocess = {
    #    'resnet': prepare_image
    # }
    # initialize the data dictionary that will be returned from the view
    data = {'success': False}
    s0 = None
    # ensure an image was properly uploaded to our endpoint
    if request.method == 'POST':
        model = request.args.get('model')
        image = request.files.get('image')
        if image and model:
            s0 = time.time()
            """
            prepare = preprocess.get(model)
            if prepare:
                image = preprocess_image(
                    request.files['image'].read(), prepare)
            else:
                image = request.files['image'].read()

            # indicate that the request was a success
            """
            image = Image.open(io.BytesIO(image.read()))
            data['success'], data['predictions'] = call_model(model, image)

    # return the data dictionary as a JSON response
    if s0:
        s1 = time.time()
        print('Results read from queue and returned in ', s1 - s0)
    return jsonify(data)


@app.route('/image/<filename>')
# @login_required
def image(filename):
    """Serializa a imagem do banco para stream HTTP."""

    filename = os.path.join(tmpdir, filename)
    image = open(filename, 'rb').read()
    return Response(response=image, mimetype='image/jpeg')


@app.route('/image_zoom/<filename>')
# @login_required
def image_zoom(filename):
    """Serializa a imagem do banco para stream HTTP."""
    filename = os.path.join(tmpdir, filename)
    image = Image.open(filename)
    success, pred_bbox = call_model('ssd', image)
    if success:
        coords = pred_bbox[0]['bbox']
        im = np.asarray(image)
        im = im[coords[0]:coords[2], coords[1]:coords[3]]
        image = Image.fromarray(im)
        img_io = io.BytesIO()
        image.save(img_io, 'JPEG', quality=70)
        img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')


@app.route('/teste', methods=['GET', 'POST'])
@csrf.exempt
# TODO: Make login in all clients and tests, then uncomment next line
# @login_required
def teste():
    """Função simplificada para teste interativo de upload de imagem"""
    result = []
    ts = ''
    print(request.method)
    print(request.form)
    import pprint
    pprint.pprint(request.files)
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
            print(file)
            # os.remove('padma/static/temp.jpg')
            ts = str(time.time())
            filename = os.path.join(tmpdir, ts + '.jpg')
            print(filename)
            with open(filename, 'wb') as temp:
                temp.write(file.read())
            # print('content', file.read())
            image = Image.open(filename)
            # success, pred_bbox = call_model('naive', image)
            success, pred_bbox = call_model('ssd', image)
            print(pred_bbox)
            if success:
                print(image.size)
                print(pred_bbox)
                result.append(json.dumps(image.size))
                result.append(json.dumps(pred_bbox))
                coords = pred_bbox[0]['bbox']
                im = np.asarray(image)
                im = im[coords[0]:coords[2], coords[1]:coords[3]]
                image = Image.fromarray(im)
                success, pred_vazio = call_model('vazio', image)
                result.append(json.dumps(pred_vazio))
                success, pred_vazio = call_model('vaziosvm', image)
                result.append(json.dumps(pred_vazio))
                success, pred_peso = call_model('peso', im)
                result.append(json.dumps(pred_peso))
                success, pred_peso = call_model('pesor', im)
                result.append(json.dumps(pred_peso))

    return render_template('teste.html', result=result, filename=ts + '.jpg')


@nav.navigation()
def mynavbar():
    items = [View('Home', 'index'),
             View('Testar modelos', 'teste'),
             ]
    if current_user.is_authenticated:
        items.append(View('Sair', 'logout'))
    return Navbar(*items)


nav.init_app(app)

app.config['DEBUG'] = os.environ.get('DEBUG', 'None') == '1'
if app.config['DEBUG'] is True:
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
app.secret_key = SECRET
app.config['SECRET_KEY'] = SECRET

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == '__main__':
    # load the function used to classify input images in a *separate*
    # thread than the one used for main classification
    """
    print('* Starting model service...')
    if platform != 'win32':
        t = Thread(target=classify_process, args=())
        t.daemon = True
        t.start()
    else:
        classify_process()
    """
    # start the web server
    print('* Starting web service...')
    app.config['DEBUG'] = True
    app.run(debug=app.config['DEBUG'], port=5002)
