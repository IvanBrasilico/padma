# JSON server
# Args
#  model, params
# Returns
#   JSON dict with model response
import pickle
import io
import json
import numpy as np
import os
import tempfile
import time
import uuid
# from base64 import b64encode
# from base64 import decodebytes
from sys import platform
from threading import Thread

from flask import (abort, Flask, flash, jsonify, redirect, render_template,
                   request, Response, send_file, url_for)
# import redis
from PIL import Image

from flask_bootstrap import Bootstrap
from flask_login import current_user, login_required
from flask_login import login_user, logout_user
from flask_nav import Nav
from flask_nav.elements import Navbar, View
from flask_wtf.csrf import CSRFProtect

from ajna_commons.flask.conf import (SECRET, DATABASE, MONGODB_URI,
                                     redisdb)
import ajna_commons.flask.login as login
from ajna_commons.flask.log import logger

from padma.models.models import Naive, Peso, Pong, Vazios
from padma.models.conteiner20e40.bbox import SSDMobileModel

from pymongo import MongoClient

PADMA_REDIS_KEY = 'PADMAKEY01'

# initialize constants used to control image spatial dimensions and
# data type
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANS = 3
IMAGE_DTYPE = 'float32'

# initialize constants used for server queuing
BATCH_SIZE = 10
SERVER_SLEEP = 0.10
CLIENT_SLEEP = 0.10
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


@app.route('/')
def index():
    if current_user.is_authenticated:
        return render_template('index.html')
    else:
        return redirect(url_for('login'))

def model_predict(model, _id, image):
    s0 = time.time()
    output = model.predict(image)
    print('preds', output)
    dump = json.dumps(output)
    redisdb.set(_id, dump)
    s1 = time.time()
    print('Images classified in ', s1 - s0)


def classify_process():
    # Load the pre-trained models, only once.
    # Then wait for incoming queries on redis
    modeldict = dict()
    print('* Loading model PONG (ping-pong test if alive) *')
    modeldict['ping'] = Pong()
    print('* Loading model Vazios...')
    modeldict['vazio'] = Vazios()
    print('* Model vazios loaded')
    print('* Loading model Peso...')
    modeldict['peso'] = Peso(linear=True)
    print('* Model peso loaded')
    print('* Loading model Peso Random Forest...')
    modeldict['pesor'] = Peso()
    print('* Model peso random forest loaded')
    print('* Loading model Naive BBox...')
    modeldict['naive'] = Naive()
    print('* Model naive bbox loaded')
    print('* Loading model SSD BBox...')
    modeldict['ssd'] = SSDMobileModel()
    print('* Model SSD bbox loaded')

    if platform != 'win32':
        # continually poll for new images to classify
        while True:
            # attempt to grab a batch of images from the database
            for model_name, model in modeldict.items():
                queue = redisdb.lrange(model_name, 0, BATCH_SIZE - 1)
                # loop over the queue
                if queue:
                    try:
                        cont = 0
                        print('Processing image classify from queue')
                        for q in queue:
                            cont += 1
                            d = pickle.loads(q)
                            # TODO: Rodar model em Thread
                            t = Thread(target=model_predict, args=([model, d['id'], d['image']]))
                            t.daemon = True
                            t.start()
                    except TypeError as err:
                        logger.debug('Erro ao recuperar modelo %s' % model_name)
                        logger.debug(str(q))
                        logger.debug(err, exc_info=True)
                    finally:
                        redisdb.ltrim(model_name, cont, -1)
            # sleep for a small amount
            time.sleep(SERVER_SLEEP)


def win32_call_model(model, image):
    """Síncrono, sem threads, para uso no desktop Windows."""
    model = model_dict[model]
    output = model.predict(d['image'])
    return True, output

def call_model(model: str, image: Image):
    """Grava requisição no redisdb e aguarda retorno até timeout.
        
        Args:
            model: string com uma chave do dicionário de modelos ativos
            image: imagem em bytes a consultar

        Returns:
            True, dict com predições em caso de sucesso
            False, dict vazio em caso de timeout
    """
    if platform == 'win32':
        return win32_call_model(model, image)
    print('Enter Sandman - sending request to queue')
    # generate an ID for the classification then add the
    # classification ID + image to the queue
    k = str(uuid.uuid4())
    d = {'id': k,
         'image': image}
    redisdb.rpush(model, pickle.dumps(d, protocol=1))
    s0 = time.time()
    # cont = 0
    while True:
        # attempt to grab the output predictions
        output = redisdb.get(k)
        # check to see if our model has classified the input image
        if output is not None:
            # cont += 1
            # add the output predictions to our data
            # dictionary so we can return it to the client
            output = output.decode('utf-8')
            predictions = json.loads(output)
            # delete the result from the database and break
            # from the polling loop
            redisdb.delete(k)
            break
        # sleep for a small amount to give the model a chance
        # to classify the input image
        time.sleep(CLIENT_SLEEP)
        s1 = time.time()
        if s1 - s0 > 5:  # Timeout
            print("Timeout!!!!")
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
                print(pred_vazio)
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
    print('* Starting model service...')
    if platform != 'win32':
        t = Thread(target=classify_process, args=())
        t.daemon = True
        t.start()
    else:
        classify_process()

    # start the web server
    print('* Starting web service...')
    app.config['DEBUG'] = True
    app.run(debug=app.config['DEBUG'], port=5002)
