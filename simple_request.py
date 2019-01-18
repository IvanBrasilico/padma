# USAGE
# python simple_request.py

# import the necessary packages
import requests
import io
import json

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://localhost:5002/predict?model=ssd"
IMAGE_PATH = "jemma.png"

# load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}
headers = {}

# payload = {"image": (io.BytesIO(image), 'image')}
# submit the request
r = requests.post(KERAS_REST_API_URL, files=payload, headers=headers)

try:
    result = r.json()
except JSONDecodeError as err:
    print(err)
    result = {'predictions': None, 'success': False}

# ensure the request was sucessful
if result["success"]:
    # loop over the predictions and display them
    for (i, res) in enumerate(result["predictions"]):
        print("{}. {}: {:.4f}".format(i + 1, res["label"],
                                      res["probability"]))

# otherwise, the request failed
else:
    print(result)
    print("Request failed")
