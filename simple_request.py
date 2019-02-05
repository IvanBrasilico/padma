# USAGE
# python simple_request.py

# import the necessary packages
import requests
import io
import json

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = 'http://localhost:5002/predict?model=Pipeline_SVC.pkl'
IMAGE_PATH = "padma/tests/stamp1.jpg"

# load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}
headers = {}

# payload = {"image": (io.BytesIO(image), 'image')}
# submit the request
r = requests.post(KERAS_REST_API_URL, files=payload, headers=headers)

try:
    result = r.json()
except json.JSONDecodeError as err:
    print(err)
    result = {'predictions': None, 'success': False}

print(result)
