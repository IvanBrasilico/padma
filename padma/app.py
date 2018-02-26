# JSON server
# Args
#  model, params
# Returns
#   JSON dict with model response
import io
import json
import os
import time
import uuid
from threading import Thread

from flask import Flask, flash, redirect, render_template, request, url_for
import redis
from PIL import Image

from flask_bootstrap import Bootstrap
from flask_login import current_user, login_required
from flask_nav import Nav
from flask_nav.elements import Navbar, View
from flask_wtf.csrf import CSRFProtect

from ajna_commons.flask.conf import (SECRET, BSON_REDIS, DATABASE, MONGODB_URI,
                                     TIMEOUT, redisdb)

from padma.models.models import Naive, Pong, Vazios
from padma.utils import base64_decode_image, base64_encode_image, prepare_image

# initialize constants used to control image spatial dimensions and
# data type
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANS = 3
IMAGE_DTYPE = 'float32'

# initialize constants used for server queuing
BATCH_SIZE = 10
SERVER_SLEEP = 0.20
CLIENT_SLEEP = 0.10


app = Flask(__name__, static_url_path='/static')
csrf = CSRFProtect(app)
Bootstrap(app)
nav = Nav()


# TODO: separate login logic
from flask import abort, redirect, render_template, request, url_for
from flask_login import login_required, login_user, logout_user
from ajna_commons.flask.login import (DBUser, authenticate, is_safe_url,
                                      login_manager)
from pymongo import MongoClient
db = MongoClient(host=MONGODB_URI)[DATABASE]

login_manager.init_app(app)
DBUser.dbsession = db


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('senha')
        registered_user = authenticate(username, password)
        if registered_user is not None:
            print('Logged in..')
            print(login_user(registered_user))
            # print('Current user ', current_user)
            next = request.args.get('next')
            if not is_safe_url(next):
                return abort(400)
            return redirect(next or url_for('index'))
        else:
            return abort(401)
    else:
        return render_template('login.html', form=request.form)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    next = request.args.get('next')
    if not is_safe_url(next):
        next = None
    return redirect(next or url_for('index'))
# 3
##############################################
# 33


def allowed_file(filename):
    """Check allowed extensions"""
    return '.' in filename and \
        filename.rsplit('.', 1)[-1].lower() in ['bson']


@app.route('/')
def index():
    if current_user.is_authenticated:
        return render_template('index.html')
    else:
        return redirect(url_for('login'))


def classify_process():
    # Load the pre-trained models, only once.
    # Then wait for incoming queries on redis
    modeldict = dict()
    # print('* Loading model RESNET...')
    # modeldict['resnet'] = ResNet(weights='imagenet')
    # print('* Model resnet loaded')
    print('* Loading model PONG (ping-pong test if alive) *')
    modeldict['ping'] = Pong()
    print('* Loading model Vazios...')
    modeldict['vazios'] = Vazios()
    print('* Model vazios loaded')
    # print('* Loading model Retina BBox...')
    # modeldict['retina'] = Retina()
    #print('* Model Retina BBox loaded')
    print('* Loading model Naive BBox...')
    modeldict['naive'] = Naive()
    print('* Model naive bbox loaded')

    # continually poll for new images to classify
    while True:
        # attempt to grab a batch of images from the database
        for model_name, model in modeldict.items():
            queue = redisdb.lrange(model_name, 0, BATCH_SIZE - 1)
            # loop over the queue
            if queue:
                s0 = time.time()
                cont = 0
                print('Processing image classify from queue')
                for q in queue:
                    cont += 1
                    # deserialize the object and obtain the input image
                    q = json.loads(q.decode('utf-8'))
                    image = base64_decode_image(q['image'], IMAGE_DTYPE,
                                                (1, IMAGE_HEIGHT, IMAGE_WIDTH,
                                                 IMAGE_CHANS))
                    preds = model.predict(image)
                    output = model.format(preds)
                    redisdb.set(q['id'], json.dumps(output))
                    # remove the set of images from our queue
                redisdb.ltrim(model_name, cont, -1)
                s1 = time.time()
                print(s1, 'Images classified in ', s1 - s0)
            # sleep for a small amount
            time.sleep(SERVER_SLEEP)


def read_model(model, image):
    print('Enter Sandman - sending request to queue')
    # generate an ID for the classification then add the
    # classification ID + image to the queue
    k = str(uuid.uuid4())
    d = {'id': k,
         'image': base64_encode_image(image)}
    redisdb.rpush(model, json.dumps(d))
    s0 = time.time()
    while True:
        # attempt to grab the output predictions
        output = redisdb.get(k)
        # check to see if our model has classified the input image
        if output is not None:
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
            redisdb.delete(k)
            return False, {}
    return True, predictions


def prepare_image(image):
    # read the image in PIL format and prepare it for classification
    image = Image.open(io.BytesIO(image))
    image = prepare_image(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    # ensure our NumPy array is C-contiguous as well,
    # otherwise we won't be able to serialize it
    image = image.copy(order='C')
    return image


@app.route('/predict', methods=['POST'])
def predict():
    # initialize the data dictionary that will be returned from the view
    data = {'success': False}
    s0 = None
    # ensure an image was properly uploaded to our endpoint
    if request.method == 'POST':
        if request.files.get('image') and model:
            model = request.args.get('model')
            s0 = time.time()
            image = prepare_image(request.files['image'].read())
            # indicate that the request was a success
            data['success'], data['predictions'] = read_model(model, image)

    # return the data dictionary as a JSON response
    if s0:
        s1 = time.time()
        print(s1, 'Results read from queue and returned in ', s1 - s0)
    return flask.jsonify(data)


@app.route('/teste', methods=['GET', 'POST'])
@csrf.exempt
@login_required
def teste():
    """Função simplificada para upload de arquivo de imagem
    """
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files.get('file')
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            return jsonify(read_model('naive', file))

    return render_template('teste.html')


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
    t = Thread(target=classify_process, args=())
    t.daemon = True
    t.start()

    # start the web server
    print('* Starting web service...')
    app.config['DEBUG'] = True
    app.run(debug=app.config['DEBUG'], port=5002)
